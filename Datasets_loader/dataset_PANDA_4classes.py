import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
import os
import glob
from skimage import io
from tqdm import tqdm


def statistic_on_label(label):
    # label is of shape N or NxC
    if type(label) is torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(label) is list:
        label = np.array(label)
    N = label.shape[0]
    if len(label.shape) == 1:
        unique_classes = np.unique(label)
        each_class_ratio = []
        for class_i in unique_classes:
            num_class_i = len(np.where(label == class_i)[0])
            each_class_ratio.append(num_class_i/N)
        print("classes: {} (N={})\nratio: {}".format(unique_classes, N, each_class_ratio))
    else:
        unique_classes = [i for i in range(label.shape[1])]
        each_class_ratio = label.sum(0) / N
        print("classes: {} (N={})\nratio: {}".format(unique_classes, N, each_class_ratio))
    return


def gather_split_PANDA_accordingCenter(root_dir='/home/xiaoyuan/Data3/PANDA_patches/patches_224x224_scale1/train', split=0.7, center_train='radboud+karolinska'):
    # PANDA doesn't provide test part, split original train to train and val part
    # Split all slide from center karolinska to train, as it doesn't provide patch-level 4 classes annos
    slide_info = pd.read_csv('/home/xiaoyuan/Data3/PANDA/prostate-cancer-grade-assessment/train.csv')
    slide_info = slide_info[['image_id', 'data_provider']].to_numpy()

    slide_from_radboud = []
    slide_from_karolinska = []
    all_slide = glob.glob(os.path.join(root_dir, '*'))
    for slide_i in all_slide:
        slide_i_name = slide_i.split('/')[-1]
        idx_ = np.where(slide_info[:, 0]==slide_i_name)[0]
        if slide_info[idx_, 1] == 'radboud':
            slide_from_radboud.append(slide_i)
        elif slide_info[idx_, 1] == 'karolinska':
            slide_from_karolinska.append(slide_i)
        else:
            raise

    slide_from_radboud = np.array(slide_from_radboud)
    slide_from_karolinska = np.array(slide_from_karolinska)
    num_slide_from_radboud = len(slide_from_radboud)
    idx_perm = np.random.permutation(num_slide_from_radboud)
    idx_train_from_radboud = idx_perm[:int(split * num_slide_from_radboud)]
    idx_test_from_radboud = idx_perm[int(split * num_slide_from_radboud):]
    if center_train == 'radboud+karolinska':
        slide_train = np.concatenate([slide_from_radboud[idx_train_from_radboud], slide_from_karolinska], axis=0)
    elif center_train == 'radboud':
        slide_train = slide_from_radboud[idx_train_from_radboud]
    elif center_train == 'karolinska':
        slide_train = slide_from_karolinska
    else:
        raise
    slide_test = slide_from_radboud[idx_test_from_radboud]
    return slide_train, slide_test


def PANDA_parse_assign_patch_label_4classes(patch_name, center='Radboud', patch_pos_threshold=0.3, patch_drop_threshold=0.5):
    # 1. parse patch name
    label = patch_name.split('label_')[1].split('_ratio')[0].split('_')
    ratio = patch_name.split('ratio_')[1].split('.jpg')[0].split('_')
    ratio = [float(i) for i in ratio]
    # 2. cast raw label to benign and cancerous according to provider
    new_label = []
    if center == 'radboud':
        for i in label:
            if i == '3':
                new_label.append('1')  # Pos 1
            elif i == '4':
                new_label.append('2')  # Pos 2
            elif i == '5':
                new_label.append('3')  # pos 3
            elif i == '1' or i == '2':
                new_label.append('0')  # Neg
            elif i == '0':
                new_label.append('-1')  # Drop
            else:
                raise
    elif center == 'karolinska':
        for i in label:
            if i == '2':
                new_label.append('1')  # in Karolinska center, patch label of gleason3,4,5 are not available
            elif i == '1':
                new_label.append('0')  # Neg
            elif i == '0':
                new_label.append('-1')  # Drop
            else:
                raise
    else:
        raise

    # 3. merge new category and corresponding area ratio
    new_label = np.array(new_label)
    ratio = np.array(ratio)
    new_label_merged = np.unique(np.array(new_label))
    new_ratio_merged = []
    for i in new_label_merged:
        idx_same_category = np.where(new_label == i)[0]
        new_ratio_merged.append(np.sum(ratio[idx_same_category]))
    new_ratio_merged = np.array(new_ratio_merged)

    # 4. assign patch label according its area
    if new_label_merged[0] == '-1' and new_ratio_merged[0] > patch_drop_threshold:  # drop patch with 50% percent area non-tissue
        patch_label = None
    else:
        patch_label = np.array([0, 0, 0, 0], dtype=int)
        for i, j in zip(new_label_merged, new_ratio_merged):
            if i == '-1':
                pass
            elif i == '0' and j >= patch_pos_threshold:
                patch_label[0] = 1
            elif i == '1' and j >= patch_pos_threshold:
                patch_label[1] = 1
            elif i == '2' and j >= patch_pos_threshold:
                patch_label[2] = 1
            elif i == '3' and j >= patch_pos_threshold:
                patch_label[3] = 1
    return patch_label


def PANDA_assign_bag_label_4classes(patches_label):
    # patches_label shape: N x 4
    assert len(patches_label.shape) == 2
    slide_label = patches_label.sum(0).clip(0, 1)
    return slide_label


def PANDA_parse_raw_gleason_label(raw_str):
    # given annos like: '0+0', '3+3', '5+4'
    if '0' in raw_str or 'negative' in raw_str:
        slide_label = np.array([1, 0, 0, 0], dtype=int)
        return slide_label
    else:
        slide_label = np.array([-1, 0, 0, 0], dtype=int)
        if '3' in raw_str:
            slide_label[1] = 1
        if '4' in raw_str:
            slide_label[2] = 1
        if '5' in raw_str:
            slide_label[3] = 1

        if slide_label[1:].sum() == 0:
            raise
        return slide_label


class PANDA_gleason_4classes(torch.utils.data.Dataset):
    def __init__(self, ds=None, transform=None, downsample=0.2, preload=False, patch_pos_threshold=0.3, patch_drop_threshold=0.5):
        self.ds = ds
        self.transform = transform
        self.downsample = downsample
        self.preload = preload
        self.patch_pos_threshold = patch_pos_threshold
        self.patch_drop_threshold = patch_drop_threshold
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        all_slides = ds
        slide_info = pd.read_csv('/home/xiaoyuan/Data3/PANDA/prostate-cancer-grade-assessment/train.csv')
        # 1.1 down sample the slides
        print("================ Down sample ================")
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        # 2.extract all available patches and build corresponding labels
        if self.preload:
            self.all_patches = np.zeros([self.num_patches, 224, 224, 3], dtype=np.uint8)
        else:
            self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        self.patch_corresponding_slide_label = []
        cnt_slide = 0
        cnt_patch = 0
        for slide_i in tqdm(all_slides, ascii=True, desc='scanning data'):
            slide_i_name = slide_i.split('/')[-1]
            slide_i_center = slide_info['data_provider'].iloc[np.where(slide_info['image_id'].to_numpy() == slide_i_name)[0][0]]
            raw_slide_label = slide_info['gleason_score'].iloc[np.where(slide_info['image_id'].to_numpy() == slide_i_name)[0][0]]
            slide_i_label = PANDA_parse_raw_gleason_label(raw_slide_label)
            for slide_i_patch_j in glob.glob(os.path.join(slide_i, "*.jpg")):
                patch_label = PANDA_parse_assign_patch_label_4classes(slide_i_patch_j.split('/')[-1],
                                                                      center=slide_i_center,
                                                                      patch_pos_threshold=self.patch_pos_threshold,
                                                                      patch_drop_threshold=self.patch_drop_threshold)
                if patch_label is None:
                    continue
                if self.preload:
                    self.all_patches[cnt_patch, :, :, :] = io.imread(slide_i_patch_j)
                else:
                    self.all_patches.append(slide_i_patch_j)
                self.patch_label.append(patch_label)
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(slide_i_name)
                self.patch_corresponding_slide_label.append(slide_i_label)
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        if not self.preload:
            self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)

        # 3. resort patches to bags and
        # (1) rebuild slide label from patch labels inside each bag
        #     [Attention] the rebuild slide label is not reliable for center karolinska,
        #     because it doesn't provide gleason 3,4,5 label,
        #     thus the rebuild label have big differences with slide label obtained above.
        # (2) revise self.patch_corresponding_slide_label, justify weather class '0' exist
        # (3) sort data into bags
        self.patch_corresponding_slide_label_rebuild = np.ones_like(self.patch_corresponding_slide_label) * -1
        self.bags = []
        self.bags_name = []
        self.bags_center = []
        self.bags_label = []
        self.bags_label_rebuild = []
        self.bags_patch_label = []
        for slide_idx_i in np.unique(self.patch_corresponding_slide_index):
            idx_from_slide_i = np.where(self.patch_corresponding_slide_index == slide_idx_i)[0]
            slide_i_name = self.patch_corresponding_slide_name[idx_from_slide_i[0]]
            slide_i_center = slide_info['data_provider'].iloc[np.where(slide_info['image_id'].to_numpy() == slide_i_name)[0][0]]
            slide_i_patches_label = self.patch_label[idx_from_slide_i]
            slide_i_label_rebuild = PANDA_assign_bag_label_4classes(slide_i_patches_label)
            self.bags.append(self.all_patches[idx_from_slide_i])
            self.bags_name.append(slide_i_name)
            self.bags_center.append(slide_i_center)
            self.bags_label.append(self.patch_corresponding_slide_label[idx_from_slide_i[0]])
            self.bags_patch_label.append(self.patch_label[idx_from_slide_i])
            self.bags_label_rebuild.append(slide_i_label_rebuild)
            self.patch_corresponding_slide_label_rebuild[idx_from_slide_i] = slide_i_label_rebuild
        self.patch_corresponding_slide_label[:, 0] = self.patch_corresponding_slide_label_rebuild[:, 0]
        # 3.do some statistics
        self.num_patches = len(self.all_patches)
        self.num_slides = len(self.bags)
        print("[DATA INFO] all patches label")
        statistic_on_label(self.patch_label)
        print("[DATA INFO] all slides label")
        statistic_on_label(self.bags_label)
        print("[DATA INFO] all slides label (rebuild version)")
        statistic_on_label(self.bags_label_rebuild)

        cnt_mismatch = 0
        for i in range(len(self.bags_label)):
            if self.bags_center[i] == 'radboud':
                if np.sum(self.bags_label[i][1:] == self.bags_label_rebuild[i][1:]) != 3:
                    cnt_mismatch = cnt_mismatch + 1
        print("Mismatch slide number in radboud: {}".format(cnt_mismatch))

        # 4. replace rebuild label with original
        self.patch_corresponding_slide_label = self.patch_corresponding_slide_label_rebuild
        self.bags_label = self.bags_label_rebuild
        del self.patch_corresponding_slide_label_rebuild
        del self.bags_label_rebuild

        # 5. post-processing, change to 3 class
        self.patch_label = self.patch_label[:, 1:]
        self.patch_corresponding_slide_label = self.patch_corresponding_slide_label[:, 1:]
        print("")

    def __getitem__(self, index):
        if self.preload:
            patch_image = self.all_patches[index]
        else:
            patch_image = io.imread(self.all_patches[index])
        patch_label = self.patch_label[index]
        patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
        patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
        patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

        patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
        return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                             patch_corresponding_slide_name], index

    def __len__(self):
        return self.num_patches


class PANDA_gleason_4classes_feat(torch.utils.data.Dataset):
    def __init__(self, ds=None, downsample=0.2, patch_pos_threshold=0.3, patch_drop_threshold=0.5, return_bag=True):
        self.ds = ds
        self.downsample = downsample
        self.patch_pos_threshold = patch_pos_threshold
        self.patch_drop_threshold = patch_drop_threshold
        self.return_bag = return_bag
        self.feat_root_dir = "/home/xiaoyuan/Data3/PANDA_patches_224x224_scal1_CLIPFeatRN50/split"

        all_slides = ds
        slide_info = pd.read_csv('/home/xiaoyuan/Data3/PANDA/prostate-cancer-grade-assessment/train.csv')
        # 1.1 down sample the slides
        print("================ Down sample ================")
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)

        # 2.extract all available patches and build corresponding labels
        self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        self.patch_corresponding_slide_label = []
        cnt_slide = 0
        cnt_patch = 0
        for slide_i in tqdm(all_slides, ascii=True, desc='scanning data'):
            slide_i_name = slide_i.split('/')[-1]
            slide_i_center = slide_info['data_provider'].iloc[np.where(slide_info['image_id'].to_numpy() == slide_i_name)[0][0]]
            raw_slide_label = slide_info['gleason_score'].iloc[np.where(slide_info['image_id'].to_numpy() == slide_i_name)[0][0]]
            slide_i_label = PANDA_parse_raw_gleason_label(raw_slide_label)

            slide_i_patches_feat = np.load(os.path.join(self.feat_root_dir, "{}_PatchFeat.npy".format(slide_i_name)))
            slide_i_patches_Name = np.load(os.path.join(self.feat_root_dir, "{}_PatchName.npy".format(slide_i_name)))

            for j in range(len(slide_i_patches_Name)):
                slide_i_patch_j = slide_i_patches_Name[j]
                patch_label = PANDA_parse_assign_patch_label_4classes(slide_i_patch_j.split('/')[-1],
                                                                      center=slide_i_center,
                                                                      patch_pos_threshold=self.patch_pos_threshold,
                                                                      patch_drop_threshold=self.patch_drop_threshold)
                if patch_label is None:
                    continue

                self.all_patches.append(slide_i_patches_feat[j])
                self.patch_label.append(patch_label)
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(slide_i_name)
                self.patch_corresponding_slide_label.append(slide_i_label)
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)

        # 3. resort patches to bags and
        # (1) rebuild slide label from patch labels inside each bag
        #     [Attention] the rebuild slide label is not reliable for center karolinska,
        #     because it doesn't provide gleason 3,4,5 label,
        #     thus the rebuild label have big differences with slide label obtained above.
        # (2) revise self.patch_corresponding_slide_label, justify weather class '0' exist
        # (3) sort data into bags
        self.patch_corresponding_slide_label_rebuild = np.ones_like(self.patch_corresponding_slide_label) * -1
        self.bags = []
        self.bags_name = []
        self.bags_center = []
        self.bags_label = []
        self.bags_label_rebuild = []
        self.bags_patch_label = []
        for slide_idx_i in np.unique(self.patch_corresponding_slide_index):
            idx_from_slide_i = np.where(self.patch_corresponding_slide_index == slide_idx_i)[0]
            if len(idx_from_slide_i) <= 1:
                continue
            slide_i_name = self.patch_corresponding_slide_name[idx_from_slide_i[0]]
            slide_i_center = slide_info['data_provider'].iloc[np.where(slide_info['image_id'].to_numpy() == slide_i_name)[0][0]]
            slide_i_patches_label = self.patch_label[idx_from_slide_i]
            slide_i_label_rebuild = PANDA_assign_bag_label_4classes(slide_i_patches_label)
            self.bags.append(self.all_patches[idx_from_slide_i])
            self.bags_name.append(slide_i_name)
            self.bags_center.append(slide_i_center)
            self.bags_label.append(self.patch_corresponding_slide_label[idx_from_slide_i[0]])
            self.bags_patch_label.append(self.patch_label[idx_from_slide_i])
            self.bags_label_rebuild.append(slide_i_label_rebuild)
            self.patch_corresponding_slide_label_rebuild[idx_from_slide_i] = slide_i_label_rebuild
        self.patch_corresponding_slide_label[:, 0] = self.patch_corresponding_slide_label_rebuild[:, 0]
        # 3.do some statistics
        self.num_patches = len(self.all_patches)
        self.num_slides = len(self.bags)
        print("[DATA INFO] all patches label")
        statistic_on_label(self.patch_label)
        print("[DATA INFO] all slides label")
        statistic_on_label(self.bags_label)
        print("[DATA INFO] all slides label (rebuild version)")
        statistic_on_label(self.bags_label_rebuild)

        cnt_mismatch = 0
        for i in range(len(self.bags_label)):
            if self.bags_center[i] == 'radboud':
                if np.sum(self.bags_label[i][1:] == self.bags_label_rebuild[i][1:]) != 3:
                    cnt_mismatch = cnt_mismatch + 1
        print("Mismatch slide number in radboud: {}".format(cnt_mismatch))

        # 4. replace rebuild label with original
        self.patch_corresponding_slide_label = self.patch_corresponding_slide_label_rebuild
        self.bags_label = self.bags_label_rebuild
        del self.patch_corresponding_slide_label_rebuild
        del self.bags_label_rebuild

        # 5. post-processing, change to 3 class
        self.patch_label = self.patch_label[:, 1:]
        self.patch_corresponding_slide_label = self.patch_corresponding_slide_label[:, 1:]
        self.bags_label = [i[1:] for i in self.bags_label]
        self.bags_patch_label = [i[:, 1:] for i in self.bags_patch_label]
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            slide_feat = self.bags[index]
            slide_label = self.bags_label[index]
            slide_patch_label = self.bags_patch_label[index]
            return slide_feat, [slide_patch_label, slide_label, index], index
        else:
            patch_image = self.all_patches[index]
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return len(self.bags_label)
        else:
            return self.num_patches


if __name__ == '__main__':
    train_ds, val_ds = gather_split_PANDA_accordingCenter(root_dir='/home/xiaoyuan/Data3/PANDA_patches/patches_224x224_scale1/train', split=0.7, center_train='radboud')
    train_ds = PANDA_gleason_4classes_feat(ds=train_ds, downsample=1.0, patch_pos_threshold=0.3, patch_drop_threshold=0.5, return_bag=True)
    val_ds = PANDA_gleason_4classes_feat(ds=val_ds, downsample=1.0, patch_pos_threshold=0.3, patch_drop_threshold=0.5, return_bag=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    for data in train_loader:
        patch_img = data[0]
        label_patch = data[1][0]
        label_bag = data[1][1]
        idx = data[-1]
    print("END")
