import numpy as np
import torch
import clip
from tqdm import tqdm
import os
import torchstain
from skimage import io
from torchvision import transforms
import cv2
from PIL import Image
import glob


def statistics_dataset(labels):
    # labels of sihape N
    num_samples = labels.shape[0]
    all_cate = np.unique(labels)
    for i in range(all_cate.shape[0]):
        num_samples_cls_i = np.sum(labels == all_cate[i])
        print("class {}: {}/{} samples, ratio:{:.4f}".format(all_cate[i], num_samples_cls_i, num_samples, num_samples_cls_i/num_samples))
    return 0


class TCGA_region_StainNorm(torch.utils.data.Dataset):
    def __init__(self, ds_data, normalizer):
        self.all_patches = ds_data

        self.normalizer = normalizer
        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])
        self.T2 = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        try:
            img = cv2.cvtColor(cv2.imread(self.all_patches[index]), cv2.COLOR_BGR2RGB)
            img, H, E = self.normalizer.normalize(I=self.T(img), stains=True)
            img = img.permute(2, 0, 1)/255
            norm_flag = 1
        except:
            print("Stain Norm failed : {}".format(self.all_patches[index]))
            img = io.imread(self.all_patches[index])
            img = self.T2(Image.fromarray(np.uint8(img), 'RGB'))
            norm_flag = 0

        return img, index, norm_flag

    def __len__(self):
        return len(self.all_patches)


class TCGA_region(torch.utils.data.Dataset):
    def __init__(self, ds_data, transform=None):
        self.all_patches = ds_data
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = io.imread(self.all_patches[index])
        img = self.transform(Image.fromarray(np.uint8(img), 'RGB'))
        return img, index

    def __len__(self):
        return len(self.all_patches)


def compute_CLIP_feat(model, dataloader):
    L = len(dataloader.dataset)
    feat_all = np.zeros([L, 1024], dtype=np.float32)
    norm_flag_all = np.ones([L], dtype=int)
    for data in tqdm(dataloader, desc='Computing features'):
        images = data[0].cuda()
        selected = data[1]
        if len(data) == 3:
            norm_flag = data[2]
        else:
            norm_flag = None
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        feat_all[selected, :] = image_features.detach().cpu().numpy()
        if norm_flag is not None:
            norm_flag_all[selected] = norm_flag.detach().cpu().numpy()
    return feat_all, norm_flag_all.astype(np.bool8)


def save_CLIP_feat_PANDA(input_data_dir, output_data_dir, stain_norm=False):
    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)

    # model, preprocess = clip.load("ViT-B/32")
    model, preprocess = clip.load("RN50")

    all_patches_path = glob.glob(os.path.join(input_data_dir, '*/*.jpg'))

    if stain_norm:
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        normalizer.HERef = torch.tensor([
            [0.5333, 0.2254],
            [0.7227, 0.8494],
            [0.4275, 0.4540]
        ])
        normalizer.maxCRef = torch.tensor([1.8521, 1.4990])
        train_ds = TCGA_region_StainNorm(all_patches_path, normalizer)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=6, drop_last=False, pin_memory=True)

        feat_train, norm_flag = compute_CLIP_feat(model, train_loader)

        np.save(os.path.join(output_data_dir, "all_patch_feat.npy"), feat_train[norm_flag])
        np.save(os.path.join(output_data_dir, "all_patch_fileName.npy"), np.array(train_ds.all_patches)[norm_flag])
    else:
        train_ds = TCGA_region(all_patches_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=6, drop_last=False, pin_memory=True)

    fDAeat_train, _ = compute_CLIP_feat(model, train_loader)

    np.save(os.path.join(output_data_dir, "all_patch_feat.npy"), feat_train)
    np.save(os.path.join(output_data_dir, "all_patch_fileName.npy"), np.array(train_ds.all_patches))
    print("END")


def post_split(clip_feat_dir='/home/xiaoyuan/Data3/PANDA_patches_224x224_scal1_CLIPFeatRN50',
               output_dir='/home/xiaoyuan/Data3/PANDA_patches_224x224_scal1_CLIPFeatRN50/split'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # post process: split all patch feat into each slide
    feat = np.load(os.path.join(clip_feat_dir, 'all_patch_feat.npy'))
    feat_name = np.load(os.path.join(clip_feat_dir, 'all_patch_fileName.npy'))

    slide_name_all = []
    for i in feat_name:
        slide_name_i = i.split('/')[-2]
        slide_name_all.append(slide_name_i)
    slide_name_all = np.array(slide_name_all)

    for slide_i in np.unique(slide_name_all):
        idx_patch_from_slide_i = np.where(slide_name_all==slide_i)[0]
        slide_i_patch_feat = feat[idx_patch_from_slide_i]
        slide_i_patch_name = feat_name[idx_patch_from_slide_i]
        np.save(os.path.join(output_dir, "{}_PatchFeat.npy".format(slide_i)), slide_i_patch_feat)
        np.save(os.path.join(output_dir, "{}_PatchName.npy".format(slide_i)), slide_i_patch_name)

    return


if __name__ == '__main__':

    # save_CLIP_feat_PANDA(input_data_dir='/home/xiaoyuan/Data3/PANDA_patches/patches_224x224_scale1/train',
    #                         output_data_dir='/home/xiaoyuan/Data3/PANDA_patches_224x224_scal1_CLIPFeatRN50/',
    #                         stain_norm=False)
    post_split(clip_feat_dir='/home/xiaoyuan/Data3/PANDA_patches_224x224_scal1_CLIPFeatRN50',
               output_dir='/home/xiaoyuan/Data3/PANDA_patches_224x224_scal1_CLIPFeatRN50/split')
    print("END")