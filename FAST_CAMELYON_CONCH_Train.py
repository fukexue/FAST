import argparse
import copy
import time

import numpy as np
import torch
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import datetime
import util
import utliz
from tqdm import tqdm
from Datasets_loader.dataset_CAMELYON16_conch import CAMELYON_16_5x_feat
# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

from models.learnable_prompt_conch import Instance_CLIP, PromptLearner
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Dropout
import os
from select_method import k_means_clustering, mini_batch_k_means

_tokenizer = get_tokenizer()


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(torch.nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model.float()


class Map_few_shot(torch.utils.data.Dataset):
    def __init__(self, ds, num_shot=-1):
        self.ds = ds
        self.num_shot = num_shot
        self.few_shot_indexes = []
        # generate few shot idx, compatible with CAMELYON_16_5x_feat
        ds_label = self.ds.slide_label_all
        cate = np.unique(ds_label)
        for cate_i in cate:
            idx_cate_i_all = np.where(ds_label==cate_i)[0]
            if self.num_shot != -1:
                idx_cate_i_few_shot = np.random.choice(idx_cate_i_all, self.num_shot, replace=False).tolist()
            else:
                idx_cate_i_few_shot = idx_cate_i_all.tolist()
            self.few_shot_indexes = self.few_shot_indexes + idx_cate_i_few_shot

        print("{}-shot dataset build".format(num_shot))

    def __getitem__(self, index):
        few_shot_idx = self.few_shot_indexes[index]
        slide_feat, label_list, index_raw = self.ds.__getitem__(few_shot_idx)
        return slide_feat, label_list, index

    def __len__(self):
        return len(self.few_shot_indexes)


class Map_BagFewShot_InstanceFewShot(torch.utils.data.Dataset):
    def __init__(self, ds, num_bag_shot=-1, num_instance_shot=-1):
        self.ds = ds
        self.num_bag_shot = num_bag_shot
        self.num_instance_shot = num_instance_shot

        # 1. generate bag few shot idx, compatible with CAMELYON_16_5x_feat
        self.bag_shot_indexes = []
        ds_label = self.ds.slide_label_all
        cate = np.unique(ds_label)
        for cate_i in cate:
            idx_cate_i_all = np.where(ds_label==cate_i)[0]
            if self.num_bag_shot != -1:
                idx_cate_i_few_shot = np.random.choice(idx_cate_i_all, self.num_bag_shot, replace=False).tolist()
            else:
                idx_cate_i_few_shot = idx_cate_i_all.tolist()
            self.bag_shot_indexes = self.bag_shot_indexes + idx_cate_i_few_shot

        # 2. generate corresponding instance few shot idx for each positive bag
        self.bag_instance_shot_indexes = []
        for bag_shot_idx in self.bag_shot_indexes:
            if self.ds.slide_label_all[bag_shot_idx] == 0:
                self.bag_instance_shot_indexes.append((bag_shot_idx, 0, [], []))
            else:
                all_pos_instance_idx = np.where(self.ds.slide_patch_label_all[bag_shot_idx] == 1)[0]
                all_neg_instance_idx = np.where(self.ds.slide_patch_label_all[bag_shot_idx] == 0)[0]

                if self.num_instance_shot > len(all_pos_instance_idx):
                    instance_shot_pos_idx = np.random.choice(all_pos_instance_idx, len(all_pos_instance_idx), replace=False).tolist()
                else:
                    instance_shot_pos_idx = np.random.choice(all_pos_instance_idx, self.num_instance_shot, replace=False).tolist()
                instance_shot_neg_idx = np.random.choice(all_neg_instance_idx, self.num_instance_shot, replace=False).tolist()

                self.bag_instance_shot_indexes.append((bag_shot_idx, 1, instance_shot_pos_idx, instance_shot_neg_idx))

        print("{}-BagShot {}-InstanceShot dataset build".format(num_bag_shot, num_instance_shot))

    def __getitem__(self, index):
        bag_few_shot_idx, bag_label, pos_instance_few_shot_idx, neg_instance_few_shot_idx = self.bag_instance_shot_indexes[index]
        slide_feat, label_list, index_raw = self.ds.__getitem__(bag_few_shot_idx)
        label_list.append(label_list[0])  # append GT instance label in PosBag for Measuring Pseudo-label Acc

        # replace instance labels from pos bag to few-shot labels
        if bag_label == 1:
            instance_few_shot_label = np.ones_like(label_list[0]) * -1
            instance_few_shot_label[pos_instance_few_shot_idx] = 1
            instance_few_shot_label[neg_instance_few_shot_idx] = 0
            label_list[0] = instance_few_shot_label
        return slide_feat, label_list, index

    def __len__(self):
        return len(self.bag_instance_shot_indexes)



def get_pathological_tissue_level_prompts(multi_templates=True):
    common_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a photo of the hard to see {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a close-up photo of a {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a photo of one {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'a low resolution photo of a {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'a photo of the large {}.',
        'a dark photo of a {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
    ]

    pathology_templates = [
        'a histopathological image of {}.',
        'a microscopic image of {} in tissue.',
        'a pathology slide showing {}.',
        'a high magnification image of {}.',
        'an immunohistochemical staining of {}.',
        'a pathology image of {} with inflammatory cells.',
        'a low magnification image of {}.',
        'a pathology image of {} with cellular atypia.',
        'a pathology image of {} with necrosis.',
        'an H&E stained image of {}.',
        'a pathology image of {} with fibrosis.',
        'a pathology image of {} with neoplastic cells.',
        'a pathology image of {} with metastasis.',
        'a pathology image of {} with infiltrating cells.',
        'a pathology image of {} with granulation tissue.',
        'an image of {} on a pathology slide.',
        'a pathology image of {} with edema.',
        'a pathology image of {} with hemorrhage.',
        'a pathology image of {} with degenerative changes.',
        'a pathology image of {} with angiogenesis.',
    ]

    knowledge_from_chatGPT = {
        "Squamous epithelium": "Flat, plate-like cells with a centrally located nucleus.",
        "Columnar epithelium": "Elongated cells with a basally located, oval-shaped nucleus.",
        "Glandular epithelium": "Cells organized in gland-like structures, secreting various substances.",
        "Adipose tissue": "Large, round cells with a thin rim of cytoplasm and a peripheral nucleus, filled with a lipid droplet.",
        "Fibrous connective tissue": "Dense arrangement of collagen fibers and fibroblast cells with elongated nuclei.",
        "Cartilage": "Chondrocytes embedded in a matrix with a basophilic appearance, arranged in lacunae.",
        "Bone tissue": "Calcified matrix with embedded osteocytes in lacunae, connected by canaliculi.",
        "Skeletal muscle": "Long, cylindrical, multinucleated cells with visible striations.",
        "Smooth muscle": "Spindle-shaped cells with a single, centrally located nucleus and no visible striations.",
        "Cardiac muscle": "Branching, striated cells with a single, centrally located nucleus and intercalated discs between cells.",
        "Neurons": "Large, star-shaped cells with a prominent, round nucleus and several processes extending from the cell body.",
        "Glial cells": "Smaller, supportive cells with a less-defined shape and a small, dark nucleus.",
        "Lymphocytes": "Small, round cells with a large, dark nucleus and a thin rim of cytoplasm.",
        "Germinal centers": "Areas of active lymphocyte proliferation and differentiation, appearing as lighter-stained regions in lymphoid tissue.",
        "Erythrocytes": "Anucleate, biconcave, disc-shaped cells.",
        "Leukocytes": "Nucleated white blood cells with various morphologies, including neutrophils, lymphocytes, and monocytes.",
        "Hepatocytes": "Large, polygonal cells with a round, centrally located nucleus and abundant cytoplasm.",
        "Sinusoids": "Vascular channels between hepatocytes, lined by endothelial cells and Kupffer cells in liver tissue.",
        "Glomeruli": "Compact, round structures composed of capillaries and specialized cells with a visible Bowman's space in kidney tissue.",
        "Tubules": "Epithelial-lined structures with various cell types, including proximal and distal tubule cells in kidney tissue.",

        "Carcinoma": "Disorganized tissue architecture, cellular atypia, and possible invasion into surrounding tissues in epithelial-derived tissues.",
        "Sarcoma": "Pleomorphic cells, high cellularity, and possible invasion into surrounding tissues in mesenchymal-derived tissues.",
        "Lymphoma": "Atypical lymphocytes, disrupted lymphoid architecture, and possible effacement of normal lymphoid structures.",
        "Leukemia": "Increased number of abnormal white blood cells in blood smears or bone marrow aspirates, with variable size and nuclear morphology.",
        "Glioma": "Atypical glial cells, increased cellularity, possible necrosis, and disruption of normal central nervous system tissue architecture.",
        "Melanoma": "Atypical melanocytes with variable size, shape, and pigmentation, cellular atypia, and invasion of surrounding tissues."
    }

    knowledge_from_chatGPT_natural = {
        "Squamous epithelium": "Thin, flat cells resembling plates, with a nucleus located in the center.",
        "Columnar epithelium": "Tall cells with an oval-shaped nucleus located toward the base.",
        "Glandular epithelium": "Cells arranged in gland-like structures, responsible for secreting various substances.",
        "Adipose tissue": "Round cells with a thin layer of cytoplasm surrounding a large lipid droplet, and a nucleus pushed to the side.",
        "Fibrous connective tissue": "Tightly packed collagen fibers with elongated nuclei in fibroblast cells.",
        "Cartilage": "Chondrocytes found within a basophilic matrix, situated in small spaces called lacunae.",
        "Bone tissue": "Hard, calcified matrix containing osteocytes in lacunae, which are connected by tiny channels called canaliculi.",
        "Skeletal muscle": "Long, tube-shaped cells with multiple nuclei and visible striations.",
        "Smooth muscle": "Spindle-shaped cells with a single, centrally located nucleus and no visible striations.",
        "Cardiac muscle": "Branched, striated cells with a single central nucleus and intercalated discs connecting the cells.",
        "Neurons": "Star-shaped cells with a large, round nucleus and various extensions coming from the cell body.",
        "Glial cells": "Smaller supporting cells with an undefined shape and a small, dark nucleus.",
        "Lymphocytes": "Tiny, round cells with a large, dark nucleus and a thin layer of cytoplasm.",
        "Erythrocytes": "Disc-shaped cells without a nucleus, featuring a biconcave shape.",
        "Leukocytes": "White blood cells with nuclei, displaying a range of shapes, including neutrophils, lymphocytes, and monocytes.",
        "Hepatocytes": "Sizeable, polygonal cells with a centrally positioned round nucleus and plenty of cytoplasm.",
        "Glomeruli": "Dense, round formations made up of capillaries and special cells, with a visible Bowman's space in kidney tissue.",
        "Tubules": "Structures lined with epithelial cells, containing various cell types like proximal and distal tubule cells in kidney tissue.",
        "Carcinoma": "Cancerous growth originating from epithelial cells, often exhibiting abnormal cell appearance and disordered tissue structure.",
        "Sarcoma": "Cancerous growth arising from mesenchymal cells, such as those found in bone, cartilage, fat, muscle, or blood vessels.",
        "Lymphoma": "Cancerous growth originating from lymphocytes or lymphoid tissue, often marked by unusual lymphocytes and disrupted lymphoid structure.",
        "Leukemia": "Cancerous growth of blood-forming tissues, characterized by a high number of abnormal white blood cells in the blood and bone marrow.",
        "Glioma": "Cancerous growth arising from glial cells in the central nervous system, often displaying abnormal cell appearance, increased cellularity, and tissue decay.",
        "Melanoma": "Cancerous growth originating from melanocytes, often marked by irregular melanocytes, abnormal cell appearance, and invasion into nearby tissues."
    }

    pathology_templates_t = 'an H&E stained image of {}.'
    common_templates_t = 'a photo of the {}.'

    if multi_templates:
        prompts_common_templates = [[common_templates_i.format(condition) for condition in knowledge_from_chatGPT.keys()] for common_templates_i in common_templates]
        prompts_pathology_template = [[pathology_templates_i.format(condition) for condition in knowledge_from_chatGPT.keys()] for pathology_templates_i in pathology_templates]
        prompts_pathology_template_withDescription = [
            [pathology_templates_i.format(tissue_type).replace(".", ", which is {}".format(tissue_description))
             for tissue_type, tissue_description in knowledge_from_chatGPT.items()]
            for pathology_templates_i in pathology_templates]

    else:
        prompts_common_templates = [common_templates_t.format(condition) for condition in knowledge_from_chatGPT.keys()]
        prompts_pathology_template = [pathology_templates_t.format(condition) for condition in knowledge_from_chatGPT.keys()]
        prompts_pathology_template_withDescription = [pathology_templates_t.format(tissue_type).replace(".", ", which is {}".format(tissue_description)) for tissue_type, tissue_description in knowledge_from_chatGPT.items()]

    prompts = [

    ]
    return prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription


def get_patch_level_prompts_forCAMELYON(tissue_type='multi'):
    common_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a photo of the hard to see {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a close-up photo of a {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a photo of one {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'a low resolution photo of a {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'a photo of the large {}.',
        'a dark photo of a {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
    ]

    pathology_templates = [
        'a histopathological image of {}.',
        'a microscopic image of {} in tissue.',
        'a pathology slide showing {}.',
        'a high magnification image of {}.',
        'an immunohistochemical staining of {}.',
        'a pathology image of {} with inflammatory cells.',
        'a low magnification image of {}.',
        'a pathology image of {} with cellular atypia.',
        'a pathology image of {} with necrosis.',
        'an H&E stained image of {}.',
        'a pathology image of {} with fibrosis.',
        'a pathology image of {} with neoplastic cells.',
        'a pathology image of {} with metastasis.',
        'a pathology image of {} with infiltrating cells.',
        'a pathology image of {} with granulation tissue.',
        'an image of {} on a pathology slide.',
        'a pathology image of {} with edema.',
        'a pathology image of {} with hemorrhage.',
        'a pathology image of {} with degenerative changes.',
        'a pathology image of {} with angiogenesis.',
    ]

    CAMELYON_tissue_types = {
        "lymphoid tissue": "Lymphoid tissue has a spongy appearance and is found in lymph nodes. It has areas with different shades, representing the cortex and medulla. The cortex is darker and more packed with cells, while the medulla has a lighter, looser arrangement.",
        "metastatic breast cancer cells": "These cancer cells in lymph nodes can appear as single cells, small groups, or even larger clusters. They typically have large, irregular nuclei and a high nucleus-to-cytoplasm ratio, making them stand out from the surrounding tissue.",
        "germinal centers": "Located within the lymph node cortex, germinal centers appear lighter in color compared to the surrounding area. They contain a mix of large and small lymphocytes, giving them a somewhat grainy appearance.",
        "sinus histiocytosis": "This condition shows up as large, irregular-shaped cells with a lot of cytoplasm. These cells, called histiocytes or macrophages, gather in the sinuses of lymph nodes and can sometimes be seen as clumps.",
        "blood vessels": "Blood vessels in lymph nodes have thin, tube-like structures with a lining of flat cells. Depending on their size, they can have slightly different appearances, but they all look like channels running through the tissue.",
        "connective tissue": "Connective tissue can be seen as a network of fibers and cells supporting the lymph nodes. It's found in the capsule and trabeculae, which are structures that help maintain the shape and organization of the lymph node.",
        "fat tissue": "Fat tissue, or adipose tissue, shows up as large, round cells with clear, empty-looking cytoplasm. These cells store energy and can often be found around lymph nodes."
    }

    CAMELYON_tissue_types_simple = {
        "normal": "normal image patch has regularly shaped cells and smaller, lighter nuclei",
        "tumor": "tumor image patch has irregular cancerous cells and larger, darker nuclei"
    }

    pathology_templates_t = 'an H&E stained image of {}.'
    common_templates_t = 'a photo of the {}.'

    if tissue_type == 'multi':
        prompts_common_templates = [[common_templates_i.format(condition) for condition in CAMELYON_tissue_types.keys()] for common_templates_i in common_templates]
        prompts_pathology_template = [[pathology_templates_i.format(condition) for condition in CAMELYON_tissue_types.keys()] for pathology_templates_i in pathology_templates]
        prompts_pathology_template_withDescription = [
            [pathology_templates_i.format(tissue_type).replace(".", ", {}".format(tissue_description))
             for tissue_type, tissue_description in CAMELYON_tissue_types.items()]
            for pathology_templates_i in pathology_templates]
    elif tissue_type == 'simple':
        prompts_common_templates = [[common_templates_i.format(condition) for condition in CAMELYON_tissue_types_simple.keys()] for common_templates_i in common_templates]
        prompts_pathology_template = [[pathology_templates_i.format(condition) for condition in CAMELYON_tissue_types_simple.keys()] for pathology_templates_i in pathology_templates]
        prompts_pathology_template_withDescription = [
            [pathology_templates_i.format(tissue_type).replace(".", ", {}".format(tissue_description))
             for tissue_type, tissue_description in CAMELYON_tissue_types_simple.items()]
            for pathology_templates_i in pathology_templates]
    else:
        print("unknown tissue type: {}".format(tissue_type))
        raise

    return prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription


def clip_classifier(prompts, conch_model):
    # with torch.no_grad():
    #     clip_weights = []
    #     for prompt_i in prompts:
    #         texts = clip.tokenize(prompt_i).cuda()
    #         class_embeddings = clip_model.encode_text(texts)
    #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #         clip_weights.append(class_embeddings)
    #     clip_weights = torch.stack(clip_weights)
    #     clip_weights = clip_weights.mean(dim=0)
    #     clip_weights /= clip_weights.norm(dim=-1, keepdim=True)
    # return clip_weights
    with torch.no_grad():
        conch_weights = []
        for prompt_i in prompts:
            tokenized_prompts = tokenize(texts=prompt_i, tokenizer=_tokenizer).cuda()
            class_embeddings = conch_model.encode_text(tokenized_prompts)
            conch_weights.append(class_embeddings.detach().cpu().numpy())
        conch_weights = torch.tensor(np.stack(conch_weights)).cuda()
        conch_weights = conch_weights.mean(dim=0)
    return conch_weights


def build_TipAdapter_cache_model(train_loader, downsample_neg_instances=1.0):
    # patches have been encoded by CLIP, we can just use them
    cache_keys = []
    cache_values = []
    for iter, (data, label, selected) in enumerate(tqdm(train_loader, desc='Building cache model')):
        instance_label = label[0].squeeze()
        data = data.squeeze()
        valid_instance_idx = torch.where(instance_label != -1)[0]
        cache_keys.append(data[valid_instance_idx])
        cache_values.append(instance_label[valid_instance_idx])
    cache_keys = torch.cat(cache_keys)
    cache_values = torch.cat(cache_values)
    cache_values = torch.nn.functional.one_hot(cache_values)

    if downsample_neg_instances > 1.0:
        print("downsample negative instances to {}".format(downsample_neg_instances))
        pos_idx = torch.where(cache_values[:, 1] == 1)[0]
        neg_idx = torch.where(cache_values[:, 0] == 1)[0]
        neg_idx = neg_idx[torch.randperm(neg_idx.shape[0])[:int(downsample_neg_instances)]]
        cache_keys = torch.cat([cache_keys[pos_idx], cache_keys[neg_idx]], dim=0)
        cache_values = torch.cat([cache_values[pos_idx], cache_values[neg_idx]], dim=0)

    print("Cache Model built:\nkey: {}\nvalue: {}".format(cache_keys.shape, cache_values.shape))
    return cache_keys, cache_values.float()


def build_MIL_Adapter_cache_model(train_loader, downsample_neg_instances=1.0):
    # patches have been encoded by CLIP, we can just use them
    # [Attention]: must be iterated in order to build the cache model under few-shot setting
    cache_keys = []
    cache_values = []
    cache_values_unmasked = []
    cache_corresponding_slide_label = []
    cache_corresponding_slide_index = []
    for iter, (data, label, selected) in enumerate(tqdm(train_loader, desc='Building cache model')):
        instance_label = label[0].squeeze()
        instance_label_unmasked = label[3].squeeze()
        data = data.squeeze()
        ## cache all labeled and unlabeled instances
        cache_keys.append(data)
        cache_values.append(instance_label)
        cache_values_unmasked.append(instance_label_unmasked)
        cache_corresponding_slide_label.append(torch.ones_like(instance_label) * label[1].squeeze())
        cache_corresponding_slide_index.append(torch.ones_like(instance_label) * label[2].squeeze())
    cache_keys = torch.cat(cache_keys)
    cache_values = torch.cat(cache_values)
    cache_values_unmasked = torch.cat(cache_values_unmasked)
    cache_corresponding_slide_label = torch.cat(cache_corresponding_slide_label)
    cache_corresponding_slide_index = torch.cat(cache_corresponding_slide_index)

    ## split cache into learnable and static parts
    idx_cache_unlabeled = torch.where(cache_values == -1)[0]
    idx_cache_labeled = torch.where(cache_values != -1)[0]

    cache_keys_unlabeled = cache_keys[idx_cache_unlabeled]
    cache_values_unlabeled = cache_values[idx_cache_unlabeled]
    cache_values_unlabeled_GT = cache_values_unmasked[idx_cache_unlabeled]
    cache_corresponding_slide_label_unlabeled = cache_corresponding_slide_label[idx_cache_unlabeled]
    cache_corresponding_slide_index_unlabeled = cache_corresponding_slide_index[idx_cache_unlabeled]

    cache_keys_labeled = cache_keys[idx_cache_labeled]
    cache_values_labeled = cache_values[idx_cache_labeled]
    cache_corresponding_slide_label_labeled = cache_corresponding_slide_label[idx_cache_labeled]
    cache_corresponding_slide_index_labeled = cache_corresponding_slide_index[idx_cache_labeled]

    if downsample_neg_instances < 0:
        # select representative neg instances
        print("[Select by cluster center] downsample negative instances to {}".format(np.abs(downsample_neg_instances)))
        pos_idx = torch.where(cache_values_labeled == 1)[0]
        neg_idx = torch.where(cache_values_labeled == 0)[0]

        # cluster_centers = k_means_clustering(cache_keys_labeled[neg_idx], num_clusters=np.abs(int(downsample_neg_instances)))
        cluster_centers = mini_batch_k_means(cache_keys_labeled[neg_idx], num_clusters=np.abs(int(downsample_neg_instances)), batch_size=1000, max_iterations=100)

        cache_keys_labeled = torch.cat([cache_keys_labeled[pos_idx], cluster_centers], dim=0)
        cache_values_labeled = torch.cat([cache_values_labeled[pos_idx], torch.zeros([np.abs(int(downsample_neg_instances))]).type(torch.int64)], dim=0)
        cache_corresponding_slide_label_labeled = torch.cat([cache_corresponding_slide_label_labeled[pos_idx], cache_corresponding_slide_label_labeled[neg_idx]], dim=0)
        cache_corresponding_slide_index_labeled = torch.cat([cache_corresponding_slide_index_labeled[pos_idx], cache_corresponding_slide_index_labeled[neg_idx]], dim=0)

    elif downsample_neg_instances > 1.0:
        print("[Random] downsample negative instances to {}".format(downsample_neg_instances))
        pos_idx = torch.where(cache_values_labeled == 1)[0]
        neg_idx = torch.where(cache_values_labeled == 0)[0]
        neg_idx = neg_idx[torch.randperm(neg_idx.shape[0])[:int(downsample_neg_instances)]]
        cache_keys_labeled = torch.cat([cache_keys_labeled[pos_idx], cache_keys_labeled[neg_idx]], dim=0)
        cache_values_labeled = torch.cat([cache_values_labeled[pos_idx], cache_values_labeled[neg_idx]], dim=0)
        cache_corresponding_slide_label_labeled = torch.cat([cache_corresponding_slide_label_labeled[pos_idx], cache_corresponding_slide_label_labeled[neg_idx]], dim=0)
        cache_corresponding_slide_index_labeled = torch.cat([cache_corresponding_slide_index_labeled[pos_idx], cache_corresponding_slide_index_labeled[neg_idx]], dim=0)

    print("Cache Model built (labeled part):\nkey: {}\nvalue: {}".format(cache_keys_labeled.shape, cache_values_labeled.shape))
    print("Cache Model built (unlabeled part):\nkey: {}\nvalue: {}".format(cache_keys_unlabeled.shape, cache_values_unlabeled.shape))
    return (cache_keys_unlabeled, cache_values_unlabeled, cache_corresponding_slide_label_unlabeled, cache_corresponding_slide_index_unlabeled,
            cache_keys_labeled,   cache_values_labeled,   cache_corresponding_slide_label_labeled,   cache_corresponding_slide_index_labeled,
            cache_values_unlabeled_GT)


def norm_logit(pred, no_softmax=False):
    pred = pred - pred.min()
    pred = pred / pred.max()
    if no_softmax:
        return pred
    pred = torch.softmax(pred, dim=-1)
    return pred


def run_tip_adapter(cache_keys, cache_values, test_features, test_labels, clip_weights, batch_size=4096):
    cache_keys = cache_keys.to(clip_weights.device)
    cache_values = cache_values.to(clip_weights.device)
    test_features = test_features.to(clip_weights.device)
    test_labels = test_labels.to(clip_weights.device)

    # Zero-shot CLIP
    if clip_weights.shape[0] != test_features.shape[1]:
        clip_weights = clip_weights.T
    if cache_keys.shape[0] != test_features.shape[1]:
        cache_keys = cache_keys.T

    clip_logits = 100. * cal_matrix_mul(test_features, clip_weights)
    clip_logits_normed = norm_logit(clip_logits)[:, 1]
    clip_auc = utliz.cal_auc(test_labels, clip_logits_normed)

    # Tip-Adapter
    init_beta = 1
    init_alpha = 1.17
    beta, alpha = init_beta, init_alpha

    num_batch = test_features.shape[0] // batch_size + 1
    output_ = torch.zeros((test_features.shape[0], 2)).cuda()
    for i in range(num_batch):
        affinity = cal_matrix_mul(test_features[i * batch_size: (i + 1) * batch_size], cache_keys, batch_size=batch_size)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        output_[i * batch_size: (i + 1) * batch_size] = cache_logits
    cache_logits = output_

    tip_logits = clip_logits + cache_logits * alpha
    tip_logits_normed = norm_logit(tip_logits)[:, 1]
    tip_auc = utliz.cal_auc(test_labels, tip_logits_normed)

    cache_logits_normed = norm_logit(cache_logits)[:, 1]
    cache_auc = utliz.cal_auc(test_labels, cache_logits_normed)

    # Search Hyperparameters
    search_scale = [7, 3]
    search_step = [100, 10]
    # search_step = [40, 4]
    # search_step = [20, 2]
    print("Seach Hyperparameters: scale: {}, step: {}".format(search_scale, search_step))

    tip_HP_auc, tip_logits_withHP = search_hp(search_scale, search_step, cache_keys, cache_values, test_features, test_labels, clip_weights)
    return cache_logits_normed, tip_logits_normed, tip_logits_withHP, cache_auc, tip_auc, tip_HP_auc


def run_tip_adapter_F(cache_keys, cache_values, test_features, test_labels, clip_weights, batch_size=4096):
    cache_keys = cache_keys.to(clip_weights.device)
    cache_values = cache_values.to(clip_weights.device)
    test_features = test_features.to(clip_weights.device)
    test_labels = test_labels.to(clip_weights.device)

    # Zero-shot CLIP
    if clip_weights.shape[0] != test_features.shape[1]:
        clip_weights = clip_weights.T
    if cache_keys.shape[0] != test_features.shape[1]:
        cache_keys = cache_keys.T

    clip_logits_test = 100. * cal_matrix_mul(test_features, clip_weights)
    clip_logits_normed_test = norm_logit(clip_logits_test)[:, 1]
    clip_auc_test = utliz.cal_auc(test_labels, clip_logits_normed_test)

    # Tip-Adapter
    cache_keys = torch.nn.Parameter(cache_keys)
    optimizer_keys = torch.optim.Adam([cache_keys], lr=0.001, eps=1e-4)
    train_features = cache_keys.T.detach()
    train_labels = cache_values
    clip_logits_train = 100. * cal_matrix_mul(train_features, clip_weights)

    init_beta = 1
    init_alpha = 1.17
    beta, alpha = init_beta, init_alpha

    for iter in range(20):

        num_batch = test_features.shape[0] // batch_size + 1
        output_train = torch.zeros((train_features.shape[0], 2)).cuda()
        for i in range(num_batch):
            affinity = cal_matrix_mul(train_features[i * batch_size: (i + 1) * batch_size], cache_keys,
                                      batch_size=batch_size)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            output_train[i * batch_size: (i + 1) * batch_size] = cache_logits
        cache_logits = output_train

        tip_logits_train = clip_logits_train + cache_logits * alpha
        tip_logits_normed_train = norm_logit(tip_logits_train)[:, 1]
        tip_auc_train = utliz.cal_auc(train_labels[:, 1], tip_logits_normed_train)

        loss = F.cross_entropy(tip_logits_train, train_labels)
        optimizer_keys.zero_grad()
        loss.backward()
        optimizer_keys.step()

        with torch.no_grad():
            num_batch = test_features.shape[0] // batch_size + 1
            output_test = torch.zeros((test_features.shape[0], 2)).cuda()
            for i in range(num_batch):
                affinity = cal_matrix_mul(test_features[i * batch_size: (i + 1) * batch_size], cache_keys, batch_size=batch_size)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                output_test[i * batch_size: (i + 1) * batch_size] = cache_logits
            cache_logits = output_test

            tip_logits_test = clip_logits_test + cache_logits * alpha
            tip_logits_normed_test = norm_logit(tip_logits_test)[:, 1]
            tip_auc_test = utliz.cal_auc(test_labels, tip_logits_normed_test)

        print('Iter: {}, Loss: {:.4f}, Train AUC: {:.4f}, Test AUC: {:.4f}'.format(iter, loss, tip_auc_train, tip_auc_test))

    # Search Hyperparameters
    search_scale = [7, 3]
    search_step = [100, 10]
    # search_step = [40, 4]
    # search_step = [20, 2]
    print("Seach Hyperparameters: scale: {}, step: {}".format(search_scale, search_step))

    with torch.no_grad():
        tip_HP_auc, tip_logits_withHP = search_hp(search_scale, search_step, cache_keys, cache_values, test_features, test_labels, clip_weights, batch_size=batch_size)
    return tip_logits_withHP, tip_HP_auc


def run_tip_adapter_key_value_learnable(cache_keys_unlabeled, cache_values_unlabeled,
                                        cache_keys_labeled, cache_values_labeled,
                                        test_features, test_labels, clip_weights,
                                        cache_values_unlabeled_GT=None,
                                        epoch=20, lr_value=0.001, lr_key=0.001, writer=None, batch_size=4096):
    cache_keys_unlabeled = cache_keys_unlabeled.to(clip_weights.device)
    cache_values_unlabeled = cache_values_unlabeled.to(clip_weights.device)
    cache_keys_labeled = cache_keys_labeled.to(clip_weights.device)
    cache_values_labeled = cache_values_labeled.to(clip_weights.device)
    test_features = test_features.to(clip_weights.device)
    test_labels = test_labels.to(clip_weights.device)

    if clip_weights.shape[0] != test_features.shape[1]:
        clip_weights = clip_weights.T
    if cache_keys_unlabeled.shape[0] != test_features.shape[1]:
        cache_keys_unlabeled = cache_keys_unlabeled.T
    if cache_keys_labeled.shape[0] != test_features.shape[1]:
        cache_keys_labeled = cache_keys_labeled.T

    # 1. build trainable cache model
    cache_values_learnable = torch.nn.Parameter(torch.ones_like(cache_values_unlabeled).float() * 0)
    cache_keys_learnable = torch.nn.Parameter(cache_keys_unlabeled)
    optimizer_values = torch.optim.Adam([cache_values_learnable], lr=lr_value)
    optimizer_keys = torch.optim.Adam([cache_keys_learnable], lr=lr_key)

    # option: optimize both labeled and unlabeled cache keys
    # cache_keys = torch.cat([cache_keys_unlabeled, cache_keys_labeled], dim=1)
    # cache_keys = torch.nn.Parameter(cache_keys)
    # optimizer_keys = torch.optim.Adam([cache_keys], lr=lr_key)

    # 2. cal clip prediction
    clip_logits_test = 100. * cal_matrix_mul(test_features, clip_weights)
    clip_logits_test = torch.softmax(clip_logits_test, dim=-1)
    clip_auc = utliz.cal_auc(test_labels, clip_logits_test[:, 1])

    # 3. training & validation
    train_features = cache_keys_labeled.T
    train_labels = cache_values_labeled
    best_cache_auc_test = 0
    best_cache_logits_test = None

    for iter in tqdm(range(epoch), desc='Training FAST'):
        cache_values = torch.cat([torch.sigmoid(cache_values_learnable), cache_values_labeled], dim=0)
        cache_values = torch.stack([1 - cache_values, cache_values], dim=1)

        cache_keys = torch.cat([cache_keys_learnable, cache_keys_labeled], dim=1)

        cache_logits_train = attention_functional_batch(train_features, cache_keys, cache_values, batch_size=batch_size)

        tip_train_auc = utliz.cal_auc(train_labels, cache_logits_train[:, 1])

        loss = F.cross_entropy(cache_logits_train, train_labels, weight=torch.Tensor([1, 20]).to(cache_logits_train.device))
        optimizer_values.zero_grad()
        optimizer_keys.zero_grad()
        loss.backward()
        optimizer_values.step()
        optimizer_keys.step()

        if iter % 500 == 0:
            with torch.no_grad():
                cache_logits_test = attention_functional_batch(test_features, cache_keys, cache_values)

                tip_test_auc = utliz.cal_auc(test_labels, cache_logits_test[:, 1])
                pseudo_label_auc = utliz.cal_auc(cache_values_unlabeled_GT, torch.sigmoid(cache_values_learnable).detach().cpu())

                if tip_test_auc > best_cache_auc_test:
                    best_cache_auc_test = tip_test_auc
                    best_cache_logits_test = cache_logits_test
            print('Iter: {}, Loss: {:.4f}, Train AUC: {:.4f}, Test AUC: {:.4f}'.format(iter, loss, tip_train_auc, tip_test_auc))
            if writer is not None:
                writer.add_scalar('train_loss', loss, iter)
                writer.add_scalar('train_instance_AUC', tip_train_auc, iter)
                writer.add_scalar('test_instance_AUC', tip_test_auc, iter)
                # writer.add_histogram('cache_values_learnable', torch.sigmoid(cache_values_learnable).detach().cpu(), iter)
                # writer.add_histogram('cache_values_learnable_P', torch.sigmoid(cache_values_learnable).detach().cpu()[torch.where(cache_values_unlabeled_GT==1)], iter)
                # writer.add_histogram('cache_values_learnable_N', torch.sigmoid(cache_values_learnable).detach().cpu()[torch.where(cache_values_unlabeled_GT==0)], iter)
                writer.add_scalar('cache_values_learnable_AUC', pseudo_label_auc, iter)

    merged_auc, merged_logits = search_hp_onlyAlpha(1, 100, clip_logits_test, best_cache_logits_test, test_labels)
    return (best_cache_logits_test, clip_logits_test, merged_logits,
            best_cache_auc_test,    clip_auc,         merged_auc)


def search_hp(search_scale, search_step, cache_keys, cache_values, features, labels, clip_weights, batch_size=4096):

    beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in range(search_step[0])]
    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]

    best_auc = 0
    best_beta, best_alpha = 0, 0
    best_logits = None

    clip_logits = 100. * cal_matrix_mul(features, clip_weights)
    for beta in tqdm(beta_list, desc='Searching Hyperparameters'):
        for alpha in alpha_list:
            # affinity = cal_matrix_mul(features, cache_keys)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

            num_batch = features.shape[0] // batch_size + 1
            output_ = torch.zeros((features.shape[0], 2)).cuda()
            for i in range(num_batch):
                affinity = cal_matrix_mul(features[i * batch_size: (i + 1) * batch_size], cache_keys, batch_size=batch_size)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                output_[i * batch_size: (i + 1) * batch_size] = cache_logits
            cache_logits = output_

            tip_logits = clip_logits + cache_logits * alpha
            # acc = cls_acc(tip_logits, labels)
            tip_logits_normed = norm_logit(tip_logits)[:, 1]
            auc = utliz.cal_auc(labels, tip_logits_normed)

            if auc > best_auc:
                # print("New best setting, beta: {:.2f}, alpha: {:.2f}; AUC: {:.4f}".format(beta, alpha, auc))
                best_auc = auc
                best_beta = beta
                best_alpha = alpha
                best_logits = tip_logits_normed

    print("After searching, setting, beta: {:.2f}, alpha: {:.2f}".format(best_beta, best_alpha))
    print("After searching, the best AUC: {:.4f}.".format(best_auc))

    return best_auc, best_logits


def search_hp_onlyAlpha(search_scale, search_step, clip_logits, cache_logits, labels):

    alpha_list = [i * (search_scale - 0.1) / search_step + 0.1 for i in range(search_step)]

    best_auc = 0
    best_alpha = 0
    best_logits = None

    for alpha in tqdm(alpha_list, desc='Searching Hyperparameters'):

        tip_logits = clip_logits * (1-alpha) + cache_logits * alpha

        tip_logits_normed = norm_logit(tip_logits[:, 1], no_softmax=True)
        auc = utliz.cal_auc(labels, tip_logits_normed)

        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha
            best_logits = tip_logits_normed

    print("After searching, setting, alpha: {:.2f}".format(best_alpha))
    print("After searching, the best AUC: {:.4f}.".format(best_auc))
    return best_auc, best_logits


def gather_instance_prediction_and_pred_bag(instance_pred, instance_corresponding_slide_index, instance_corresponding_slide_label):
    if type(instance_pred) is torch.Tensor:
        instance_pred = instance_pred.detach().cpu().numpy()
    if len(instance_pred.shape) == 2:
        instance_pred = instance_pred[:, 1]

    bag_label_gt = []
    bag_label_pred = []
    for slide_index_i in np.unique(instance_corresponding_slide_index):
        instance_idx_from_slide_i = np.where(instance_corresponding_slide_index == slide_index_i)[0]
        instance_pred_from_slide_i = instance_pred[instance_idx_from_slide_i]
        if instance_corresponding_slide_label[instance_idx_from_slide_i].max() != instance_corresponding_slide_label[instance_idx_from_slide_i].min():
            print("Warning: slide {} contains both positive and negative instances".format(slide_index_i))
            raise
        pred_slide_i = np.max(instance_pred_from_slide_i)
        gt_slide_i = instance_corresponding_slide_label[instance_idx_from_slide_i[0]]
        bag_label_gt.append(gt_slide_i)
        bag_label_pred.append(pred_slide_i)
    bag_label_gt = np.array(bag_label_gt)
    bag_label_pred = np.array(bag_label_pred)
    bag_auc = utliz.cal_auc(bag_label_gt, bag_label_pred)
    return bag_auc


def cal_matrix_mul(matrix1, matrix2, batch_size=4096):
    # matrix1: [N, D]
    # matrix2: [D, M]
    # return: [N, M]
    if batch_size == -1:
        return matrix1 @ matrix2
    else:
        N, D = matrix1.shape
        D, M = matrix2.shape
        num_batch = matrix1.shape[0] // batch_size + 1
        output = torch.zeros([N, M]).to(matrix1.device).type(matrix1.dtype)
        for i in range(num_batch):
            output[i * batch_size: (i + 1) * batch_size] = matrix1[i * batch_size: (i + 1) * batch_size] @ matrix2
        return output


def attention_functional_batch(query, key, value, batch_size=4096):
    # query: [N, D]
    # key: [K, D]
    # value: [K, C]
    if key.shape[1] != query.shape[1]:
        key = key.T
    N, D = query.shape
    K, C = value.shape

    if batch_size == -1:
        output = torch.softmax(query @ key.t(), dim=-1) @ value
    else:
        output = torch.zeros([N, C]).to(query.device).type(query.dtype)
        num_batch = N // batch_size + 1
        for i in range(num_batch):
            output[i * batch_size: (i + 1) * batch_size] = torch.softmax(query[i * batch_size: (i + 1) * batch_size] @ key.t(), dim=-1) @ value
    return output


class Optimizer_TextBranch:
    def __init__(self, model, train_loader, test_loader, optimizer,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10

    def optimize(self):
        for epoch in range(self.num_epoch):
            self.train_one_epoch(epoch)
            if (epoch % 10 == 0) or (epoch == self.num_epoch-1):
                self.test(epoch)
        return 0

    def train_one_epoch(self, epoch):
        self.model.train()
        loader = self.train_loader

        patch_label = []
        patch_pred = []
        # for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} training'.format(epoch))):
        for iter, (data, label, selected) in enumerate(loader):
            data = data.to(self.dev)
            label = label.to(self.dev)

            prediction = self.model(data)
            prediction = torch.softmax(prediction, 1)
            loss = torch.mean(-1. * (label * torch.log(prediction[:, 1]+1e-5) + (1. - label) * torch.log(1. - prediction[:, 1]+1e-5)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            patch_label.append(label.detach().cpu().numpy())
            patch_pred.append(prediction.detach().cpu().numpy()[:, 1])

            niter = epoch * len(loader) + iter
            self.writer.add_scalar('train_loss', loss.item(), niter)
            # print('\ntrain_loss: {:.4f}'.format(loss.item()))

        patch_label = np.concatenate(patch_label)
        patch_pred = np.concatenate(patch_pred)

        instance_auc = utliz.cal_auc(patch_label.reshape(-1), patch_pred.reshape(-1))
        print("Epoch{} train_instance_AUC: {:.4f}".format(epoch, instance_auc))
        self.writer.add_scalar('train_instance_AUC', instance_auc, epoch)

        return 0

    def test(self, epoch):
        self.model.eval()
        loader = self.test_loader

        patch_label = []
        patch_pred = []
        # for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} testing'.format(epoch))):
        for iter, (data, label, selected) in enumerate(loader):
            data = data.to(self.dev)
            label = label.to(self.dev)
            with torch.no_grad():
                prediction = self.model(data)
                prediction = torch.softmax(prediction, 1)

            patch_label.append(label.detach().cpu().numpy())
            patch_pred.append(prediction.detach().cpu().numpy()[:, 1])

        patch_label = np.concatenate(patch_label)
        patch_pred = np.concatenate(patch_pred)

        instance_auc = utliz.cal_auc(patch_label.reshape(-1), patch_pred.reshape(-1))
        print("\nEpoch{} test_instance_AUC: {:.4f}".format(epoch, instance_auc))
        self.writer.add_scalar('test_instance_AUC', instance_auc, epoch)
        return 0


class simple_instance_dataset(torch.utils.data.Dataset):
    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

    def __getitem__(self, index):
        return self.feat[index], self.label[index], index

    def __len__(self):
        return len(self.label)


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=20000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=4096, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr_keys', default=0.001, type=float, help='initial learning rate of learnable keys (default: 0.001)')
    parser.add_argument('--lr_values', default=0.01, type=float, help='initial learning rate of learnable values (default: 0.001)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='Debug_MILCLIP', type=str, help='name for tensorboardX')
    parser.add_argument('--save_intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--num_bag_shot', default=16, type=int, help='num of bag few shot')
    parser.add_argument('--num_instance_shot', default=16, type=int, help='num of instance few shot')

    parser.add_argument('--downsample_neg_instances', default=-2000, type=float, help='downsample neg instance when building cache model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_Bs{}_LrKey{}_LrValue{}_{}BagShot{}InstShot_CacheModelDownsample{}".format(
               args.seed, args.batch_size, args.lr_keys, args.lr_values, args.num_bag_shot, args.num_instance_shot,
               args.downsample_neg_instances
           )
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=args.seed, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))

    # Setup loaders
    train_ds_return_bag = CAMELYON_16_5x_feat(split='train', return_bag=True, feat="RN50")
    train_ds_return_bag = Map_BagFewShot_InstanceFewShot(train_ds_return_bag, num_bag_shot=args.num_bag_shot, num_instance_shot=args.num_instance_shot)
    val_ds_return_bag = CAMELYON_16_5x_feat(split='test', return_bag=True, feat="RN50")

    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)

    # Build cache model through iterating over few-shot training set
    cache_keys_unlabeled, cache_values_unlabeled, cache_corresponding_slide_label_unlabeled, cache_corresponding_slide_index_unlabeled, \
        cache_keys_labeled, cache_values_labeled, cache_corresponding_slide_label_labeled, cache_corresponding_slide_index_labeled, cache_values_unlabeled_GT = \
        build_MIL_Adapter_cache_model(train_loader_bag, downsample_neg_instances=args.downsample_neg_instances)
    test_features, test_labels = torch.from_numpy(val_ds_return_bag.all_patches).cuda(), torch.from_numpy(val_ds_return_bag.patch_label).cuda()

    # Setup CLIP model and prompts
    # clip_model = load_clip_to_cpu(backbone_name='RN50').cuda()
    model_cfg = 'conch_ViT-B-16'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = './conch/pytorch_model.bin'
    clip_model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)

    prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription = get_patch_level_prompts_forCAMELYON(tissue_type='simple')
    classifer_common_templates = clip_classifier(prompts_common_templates, clip_model)
    classifer_pathology_template = clip_classifier(prompts_pathology_template, clip_model)
    classifer_pathology_template_withDescription = clip_classifier(prompts_pathology_template_withDescription, clip_model)

    # Build logger
    print(name, flush=True)
    writer = SummaryWriter(log_dir=os.path.join("./runs_CAMELYON_withTextLearnable", name))

    ## Build learnable prompt and optimize it
    learnablePrompt_ctx_init = [
        "An image of evenly spaced, regular cells, suggesting healthy tissue. * * * * * * * * * *",
        "An image of disorganized, dense, and irregular cells, suggesting cancer. * * * * * * * * * *"
    ]
    instance_prompt_learner = PromptLearner(n_ctx=16, ctx_init=learnablePrompt_ctx_init, all_ctx_trainable=True, csc=True,
                                            classnames=["normal", "tumor"],
                                            clip_model='RN50', p_drop_out=0.0)
    model_text = Instance_CLIP(instance_prompt_learner).to('cuda:0')
    for param in model_text.text_encoder.parameters():
        param.requires_grad = False
    optimizer_text_branch = torch.optim.SGD(model_text.prompt_learner.parameters(), lr=0.001)

    train_ds_fewShotInstance = simple_instance_dataset(feat=cache_keys_labeled, label=cache_values_labeled)
    val_ds_allInstance = simple_instance_dataset(feat=test_features, label=test_labels)
    train_loader_fewShotInstance = torch.utils.data.DataLoader(train_ds_fewShotInstance, batch_size=1024, shuffle=True, num_workers=0, drop_last=False)
    val_loader_allInstance = torch.utils.data.DataLoader(val_ds_allInstance, batch_size=1024, shuffle=False, num_workers=0, drop_last=False)

    optimizer = Optimizer_TextBranch(model=model_text, train_loader=train_loader_fewShotInstance, test_loader=val_loader_allInstance,
                          optimizer=optimizer_text_branch,
                          writer=writer, num_epoch=50,
                          dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer.optimize()
    classifer_pathology_after_optimizing = model_text.get_classifier().detach()
    del model_text, instance_prompt_learner

    ## FAST: 1. unlabeled instance values learnable; 2. only unlabeled instance keys learnable
    start_time = time.time()
    cache_logits, clip_logits, merged_logits, cache_instanceAUC, clip_instanceAUC, merge_instanceAUC = run_tip_adapter_key_value_learnable(
        cache_keys_unlabeled, cache_values_unlabeled, cache_keys_labeled, cache_values_labeled,
        test_features, test_labels,
        clip_weights=classifer_pathology_after_optimizing,
        cache_values_unlabeled_GT=cache_values_unlabeled_GT,
        epoch=args.epochs, lr_value=args.lr_values, lr_key=args.lr_keys, writer=writer, batch_size=args.batch_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The training time is {elapsed_time} seconds.")
    cache_bagAUC = gather_instance_prediction_and_pred_bag(cache_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    clip_bagAUC = gather_instance_prediction_and_pred_bag(clip_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    merged_bagAUC = gather_instance_prediction_and_pred_bag(merged_logits, val_ds_return_bag.patch_corresponding_slide_index, val_ds_return_bag.patch_corresponding_slide_label)
    print("cache_InsAUC: {:.4f}, \nCLIP_InsAUC: {:.4f}, \nmerge_InsAUC: {:.4f}, \n".format(cache_instanceAUC, clip_instanceAUC, merge_instanceAUC))
    print("cache_BagAUC: {:.4f}, \nCLIP_BagAUC: {:.4f}, \nmerge_BagAUC: {:.4f}, \n".format(cache_bagAUC, clip_bagAUC, merged_bagAUC))
    writer.add_scalar('cache_InsAUC', cache_instanceAUC, 0)
    writer.add_scalar('merge_InsAUC', merge_instanceAUC, 0)
    writer.add_scalar('clip_InsAUC', clip_instanceAUC, 0)
    writer.add_scalar('cache_BagAUC', cache_bagAUC, 0)
    writer.add_scalar('merge_BagAUC', merged_bagAUC, 0)
    writer.add_scalar('clip_BagAUC', clip_bagAUC, 0)
    print()

