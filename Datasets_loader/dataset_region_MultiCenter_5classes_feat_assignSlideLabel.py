import os, sys
import random

import numpy as np
import pandas as pd
import torch
# import openslide
import glob
from skimage import io
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
from random import sample


def course_match(index_bing_li_hao, name_i):
    # new_index = np.zeros_like(index_bing_li_hao)
    # for i in range(len(index_bing_li_hao)):
    #     new_name_i = index_bing_li_hao[i].replace('S2013', '13')
    #     new_name_i = new_name_i.replace('-0', '-')
    #     new_index[i] = new_name_i
    # idx_match = np.where(new_index == name_i)[0]

    name_components = name_i.split('-')
    if len(name_components) == 1:
        return []

    if name_components[0] in ['11', '12', '13', '14']:
        name_components[0] = name_components[0].replace('11', 'S2011')
        name_components[0] = name_components[0].replace('12', 'S2012')
        name_components[0] = name_components[0].replace('13', 'S2013')
        name_components[0] = name_components[0].replace('14', 'S2014')
    elif name_components[0] in ['2011', '2012', '2013', '2014']:
        name_components[0] = name_components[0].replace('2011', 'S2011')
        name_components[0] = name_components[0].replace('2012', 'S2012')
        name_components[0] = name_components[0].replace('2013', 'S2013')
        name_components[0] = name_components[0].replace('2014', 'S2014')

    if name_components[1] in ['001', '005', '1', '2']:
        new_name_i = name_components[0]
    elif len(name_components[1]) <= 5:
        name_components[1] = "{:0>5}".format(name_components[1])
        new_name_i = "-".join(name_components)
    else:
        print("[Unexpected Name] {}".format(name_i))
        raise

    idx_match = np.where(index_bing_li_hao == new_name_i)[0]

    # re-match
    if len(idx_match) == 0:
        name_components = name_i.split('-')
        if name_components[0] in ['11', '12', '13', '14']:
            name_components[0] = name_components[0].replace('11', '11S')
            name_components[0] = name_components[0].replace('12', '12S')
            name_components[0] = name_components[0].replace('13', '12S')
            name_components[0] = name_components[0].replace('14', '14S')
        elif name_components[0] in ['2011', '2012', '2013', '2014']:
            name_components[0] = name_components[0].replace('2011', '11S')
            name_components[0] = name_components[0].replace('2012', '12S')
            name_components[0] = name_components[0].replace('2013', '13S')
            name_components[0] = name_components[0].replace('2014', '14S')

        if name_components[1] in ['001', '005', '1', '2']:
            new_name_i = name_components[0]
        elif len(name_components[1]) <= 5:
            name_components[1] = "{:0>5}".format(name_components[1])
            new_name_i = "".join(name_components)
        else:
            print("[Unexpected Name] {}".format(name_i))
            raise

        idx_match = np.where(index_bing_li_hao == new_name_i)[0]

    return idx_match


def match_pathology_label_ZS(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_yi_ji_hao = slide_info['医技号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = slide_name_i_raw.split(' ')[0]
        slide_name_i = slide_name_i.split('(')[0]
        slide_name_i = slide_name_i.replace('s', 'S')
        if "_" in slide_name_i:
            slide_name_i = slide_name_i.split('_')[0]
            # print("[Replace Name] {} --> {}".format(slide_name_i_raw, slide_name_i))
        if len(slide_name_i.split('-')) >= 3:
            slide_name_i = slide_name_i.split('-')[0]
            # print("[Replace Name] {} --> {}".format(slide_name_i_raw, slide_name_i))

        if '-' in slide_name_i:
            if slide_name_i.split('-')[1] in ['a1', 'a2', 'a3', 'a4', 'a5', 'a10', 'a15', 'a16', 'a17',
                                              '1', '2', '3', '4', 'N1']:
                slide_name_i = slide_name_i.split('-')[0]

        idx_match_0 = np.where(index_zhu_yuan_hao == slide_name_i)[0]
        idx_match_1 = np.where(index_yi_ji_hao == slide_name_i)[0]
        idx_match_2 = np.where(index_bing_li_hao == slide_name_i)[0]

        if len(idx_match_0) + len(idx_match_1) + len(idx_match_2) == 0:
            # print("[Course Matching Slide] {}".format(slide_name_i))
            idx_match_3 = course_match(index_bing_li_hao, slide_name_i)
            if len(idx_match_3) != 1 :
                print("[Slide Not Found or Found Twice] {}".format(slide_name_i))
                course_match(index_bing_li_hao, slide_name_i)
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 11][idx_match_3[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 13][idx_match_3[0]]
                slide_prognosis = slide_info.to_numpy()[:, 24:30][idx_match_3[0]]
        else:
            idx_match_final = np.unique(np.concatenate([idx_match_0, idx_match_1, idx_match_2]))
            if len(idx_match_final) != 1:
                print("[Slide Found Twice] {}".format(slide_name_i))
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 11][idx_match_final[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 13][idx_match_final[0]]
                slide_prognosis = slide_info.to_numpy()[:, 24:30][idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all


def match_pathology_label_TCGA(slide_info, slide_name):
    index_bianhao = slide_info['编号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = "-".join(slide_name_i_raw.split('-')[3:6])

        idx_match_final = np.where(index_bianhao == slide_name_i)[0]

        if len(idx_match_final) != 1:
            print("[Slide Found Twice] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 2][idx_match_final[0]]
            slide_nuclearLevel_label = slide_name_i_raw.split('-')[2]
            slide_prognosis = slide_info.to_numpy()[:, 29:35][idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all


def match_pathology_label_CCRCC(slide_info, slide_name):
    index_bianhao = slide_info['编号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = "-".join(slide_name_i_raw.split('-')[0:2])

        idx_match_final = np.where(index_bianhao == slide_name_i)[0]

        if len(idx_match_final) != 1:
            print("[Slide Found Twice] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 1][idx_match_final[0]]
            slide_nuclearLevel_label = -1
            slide_prognosis = slide_info.to_numpy()[:, -6:][idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all


def match_pathology_label_xiamen(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    for slide_name_i_raw in slide_name:
        if '-' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split('-')[0]
        elif ' ' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split(' ')[0]
        else:
            slide_name_i = slide_name_i_raw

        idx_match_0 = np.where(index_zhu_yuan_hao == slide_name_i)[0]
        idx_match_1 = np.where(index_bing_li_hao == slide_name_i)[0]
        idx_match_final = np.unique(np.concatenate([idx_match_0, idx_match_1]))

        if len(idx_match_final) != 1:
            print("[Slide Found Twice] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_final[0]]
            slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, None


def match_pathology_label_zhangye(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    for slide_name_i_raw in slide_name:
        if '-' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split('-')[0]
        elif ' ' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split(' ')[0]
        else:
            slide_name_i = slide_name_i_raw

        idx_match_0 = np.where(index_zhu_yuan_hao == slide_name_i)[0]
        idx_match_1 = np.where(index_bing_li_hao == slide_name_i)[0]
        idx_match_final = np.unique(np.concatenate([idx_match_0, idx_match_1]))

        if len(idx_match_final) != 1:
            print("[Slide Found Twice] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 8][idx_match_final[0]]
            slide_nuclearLevel_label = slide_info.to_numpy()[:, 9][idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, None


def match_pathology_label_huadong(slide_info, slide_name):
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = slide_name_i_raw.split(' ')[0]
        slide_name_i = slide_name_i.replace('_', '')
        if len(slide_name_i.split('-')[1]) < 5:
            slide_name_i = slide_name_i.split('-')[0] + '-' + slide_name_i.split('-')[1].zfill(5)

        idx_match = np.where(index_bing_li_hao == slide_name_i)[0]

        if len(idx_match) == 0:
            print("[Slide Not Found] {}".format(slide_name_i))
            course_match(index_bing_li_hao, slide_name_i)
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
        else:
            idx_match_final = np.unique(np.concatenate([idx_match]))
            if len(idx_match_final) != 1:
                print("[Slide Found Twice] {}".format(slide_name_i))
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 9][idx_match_final[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 11][idx_match_final[0]]
                slide_prognosis = slide_info.to_numpy()[:, 28:34][idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all


def statistics_dataset(labels):
    # labels of sihape N
    if type(labels) is list:
        labels = np.array(labels)
    num_samples = labels.shape[0]
    all_cate = np.unique(labels)
    for i in range(all_cate.shape[0]):
        num_samples_cls_i = np.sum(labels == all_cate[i])
        print("class {}: {}/{} samples, ratio:{:.4f}".format(all_cate[i], num_samples_cls_i, num_samples, num_samples_cls_i/num_samples))
    return


def parse_fileName(fileName):
    if type(fileName) is np.ndarray:
        label = []
        for i in range(len(fileName)):
            label.append(int(fileName[i].split('/')[-1].split('_')[-1][5:-4])-1)
        label = np.array(label)
    else:
        label = int(fileName.split('/')[-1].split('_')[-1][5:-4])-1
    return label


def convert_patho_label(raw_list):
    # map_dict = {
    #     '-1': -1,
    #     '1.1': 1,  '1.2': 2,
    #     '2.1': 3,  '2.2': 10, '2.3': 23,
    #     '3.1': 4,  '3.2': 5,  '3.3': 6,
    #     '4.1': 7,  '4.2': 21,
    #     '5.1': 8,  '5.2': 9,  '5.3': 22, '5.4': 20, '5.5': 28, '5.6': 13,
    #     '6.1': 14, '6.2': 15, '6.3': 16, '6.4': 17, '6.5': 18,
    #
    #     '7.1': 19, '7.2': 11, '7.3': 46,
    #     'b': 47,
    #     'c': 48,
    #     'd1': 24, 'd2': 25, 'd3': 26, 'd4': 27, 'd5': 12, 'd6': 29, 'd7': 30, 'd8': 31, 'd9': 32,
    #     'e1': 33, 'e2': 34,
    #     'f1': 35, 'f2': 36,
    #     'g': 37,
    #     'h1': 38, 'h2': 39, 'h3': 40, 'h4': 41,
    #     'i': 42,
    #     'j': 43,
    #     'k': 44,
    #     'l': 45,
    #     'z': -1
    # }

    # map_dict = {
    #     '-1': -1,
    #     '1.1': 0,  '1.2': 1,
    #     '2.1': 2,  '2.2': 3, '2.3': 4,
    #     '3.1': 5,  '3.2': 6, '3.3': 7,
    #     '4.1': 8,  '4.2': 9,
    #     '5.1': 10, '5.2': 11, '5.3': 12, '5.4': 13, '5.5': 14, '5.6': 15,
    #     '6.1': 16, '6.2': 17, '6.3': 18, '6.4': 19, '6.5': 20,
    #
    #     '7.1': -1, '7.2': -1, '7.3': -1,
    #     'b': -1,
    #     'c': -1,
    #     'd1': -1, 'd2': -1, 'd3': -1, 'd4': -1, 'd5': -1, 'd6': -1, 'd7': -1, 'd8': -1, 'd9': -1,
    #     'e1': -1, 'e2': -1,
    #     'f1': -1, 'f2': -1,
    #     'g': -1,
    #     'h1': -1, 'h2': -1, 'h3': -1, 'h4': -1,
    #     'i': -1,
    #     'j': -1,
    #     'k': -1,
    #     'l': -1,
    #     'z': -1
    # }

    map_dict = {
        '-1': -1,
        '1.1': 0,  '1.2': 1, '1.3': -1,
        '2.1': 2,  '2.2': 3, '2.3': 4,
        '3.1': 5,  '3.2': 6, '3.3': -1,
        '4.1': 8,  '4.2': 9,
        '5.1': 10, '5.2': 11, '5.3': -1, '5.4': 13, '5.5': 14, '5.6': 15,
        '6.1': 16, '6.2': 17, '6.3': 18, '6.4': 19, '6.5': 12,

        '7.1': -1, '7.2': -1, '7.3': -1,
        'b': -1,
        'c': -1,
        'd1': -1, 'd2': -1, 'd3': -1, 'd4': -1, 'd5': -1, 'd6': -1, 'd7': -1, 'd8': -1, 'd9': -1,
        'e1': -1, 'e2': -1,
        'f1': -1, 'f2': -1,
        'g': -1,
        'h1': -1, 'h2': -1, 'h3': -1, 'h4': -1,
        'i': -1,
        'j': -1,
        'k': -1,
        'l': -1,
        'z': -1
    }
    new_list = np.zeros_like(raw_list, dtype=int)
    for i in range(len(raw_list)):
        new_list[i] = map_dict[str(raw_list[i])]
    return new_list


def convert_nuclearLevel_label(raw_list):
    map_dict = {
        '-1': -1,
        '1': 1,   '2': 2,   '3': 3,   '4': 4,
        '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4,

        '1,2': 2, '2,3': 3, '2,4': 4, '3,4': 4,
        '1,2,4': 4, '2,3,4': 4,

        'nan': -1, 'NA': -1
    }
    new_list = np.zeros_like(raw_list, dtype=int)
    for i in range(len(raw_list)):
        new_list[i] = map_dict[str(raw_list[i])]
    return new_list


def convert_prognosis_label(raw_list):
    new_list = []
    for i in range(len(raw_list)):
        if type(raw_list[i]) is int:
            if raw_list[i] == -1:
                new_list.append(np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32))
        else:
            new_list.append(np.nan_to_num(raw_list[i].astype(np.float32), nan=-1))
    return new_list


def get_sort_one_center(center_dir, downsample=1.0, shuffle=True, slide_patho_anno_path=None, center='ZS'):
    # 1. load CLIP feat
    all_patches_file = np.load(os.path.join(center_dir, "all_patch_feat.npy"))
    all_patches_fileName = np.load(os.path.join(center_dir, "all_patch_fileName.npy"))

    # 1.1 remove nan feat
    t_ = all_patches_file.max(axis=1)
    idx_nan = np.isnan(t_)
    print("remove {} nan feat vector".format(idx_nan.sum()))
    all_patches_file = all_patches_file[~idx_nan]
    all_patches_fileName = all_patches_fileName[~idx_nan]

    # 2. sort patches into slides
    all_patche_corresponding_slideName = all_patches_fileName.tolist()
    for i in range(len(all_patche_corresponding_slideName)):
        all_patche_corresponding_slideName[i] = all_patche_corresponding_slideName[i].split('/')[-3]
    all_patche_corresponding_slideName = np.array(all_patche_corresponding_slideName)
    unique_slideName = np.unique(all_patche_corresponding_slideName)

    if downsample < 1.0:
        unique_slideName = np.random.choice(unique_slideName, int(len(unique_slideName)*downsample), replace=False)
    if shuffle:
        unique_slideName = np.random.permutation(unique_slideName)

    slide_patch_feat = []
    slide_patch_label = []
    slide_patch_fileName = []
    for slide_i in unique_slideName:
        idx_from_slide_i = np.where(all_patche_corresponding_slideName == slide_i)
        slide_patch_feat.append(all_patches_file[idx_from_slide_i])
        slide_patch_label.append(parse_fileName(all_patches_fileName[idx_from_slide_i]))
        slide_patch_fileName.append(all_patches_fileName[idx_from_slide_i])

    slide_info = pd.read_excel(slide_patho_anno_path)

    if center == 'ZS':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis = match_pathology_label_ZS(slide_info, [slide_patch_fileName[i][0].split('/')[9] for i in range(len(slide_patch_fileName))])
    elif center == 'TCGA':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis = match_pathology_label_TCGA(slide_info, [slide_patch_fileName[i][0].split('/')[9] for i in range(len(slide_patch_fileName))])
    elif center == 'CCRCC':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis = match_pathology_label_CCRCC(slide_info, [slide_patch_fileName[i][0].split('/')[9] for i in range(len(slide_patch_fileName))])
    elif center == 'xiamen':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis = match_pathology_label_xiamen(slide_info, [slide_patch_fileName[i][0].split('/')[9] for i in range(len(slide_patch_fileName))])
    elif center == 'zhangye':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis = match_pathology_label_zhangye(slide_info, [slide_patch_fileName[i][0].split('/')[9] for i in range(len(slide_patch_fileName))])
    elif center == 'huadong':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis = match_pathology_label_huadong(slide_info, [slide_patch_fileName[i][0].split('/')[9] for i in range(len(slide_patch_fileName))])

    else:
        raise

    slide_patho_label = convert_patho_label(slide_patho_label).tolist()
    slide_nuclearLevel_label = convert_nuclearLevel_label(slide_nuclearLevel_label).tolist()

    if slide_prognosis is None:
        slide_prognosis_label = [np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32) for i in range(len(slide_patho_label))]
    else:
        slide_prognosis_label = convert_prognosis_label(slide_prognosis)
        # slide_prognosis_label = slide_prognosis
    return (slide_patch_feat, slide_patch_label,
            unique_slideName.tolist(), slide_patch_fileName,
            slide_patho_label, slide_nuclearLevel_label, slide_prognosis_label)


def get_train_test_ds_MultiCenter_region(
        data_root='/home/xiaoyuan/Data3/dataset_Renal_Inhouse/2023_1209/MultiCenterPatches_5Classes_sameMpp_CLIPfeat_full/CLIP_ViTB32_feat_StainNorm',
        downsample=1.0):

    slide_patch_feat_HUADONG, slide_patch_label_HUADONG, slide_fileName_HUADONG, slide_patch_fileName_HUADONG, slide_patho_label_HUADONG, slide_nuclearLevel_label_HUADONG, slide_prognosis_HUADONG = get_sort_one_center(os.path.join(data_root, "output_hua_dong_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/华东肾癌整理10-1.xlsx", center='huadong')
    print("xia_men Load")

    slide_patch_feat_XIAMEN, slide_patch_label_XIAMEN, slide_fileName_XIAMEN, slide_patch_fileName_XIAMEN, slide_patho_label_XIAMEN, slide_nuclearLevel_label_XIAMEN, slide_prognosis_XIAMEN = get_sort_one_center(os.path.join(data_root, "output_xia_men_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/厦门病理.xlsx", center='xiamen')
    print("xia_men Load")

    slide_patch_feat_ZHANGYE, slide_patch_label_ZHANGYE, slide_fileName_ZHANGYE, slide_patch_fileName_ZHANGYE, slide_patho_label_ZHANGYE, slide_nuclearLevel_label_ZHANGYE, slide_prognosis_ZHANGYE = get_sort_one_center(os.path.join(data_root, "output_zhang_ye_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/张掖队列.xlsx", center='zhangye')
    print("zhang_ye Load")

    slide_patch_feat_TCGAKICH, slide_patch_label_TCGAKICH, slide_fileName_TCGAKICH, slide_patch_fileName_TCGAKICH, slide_patho_label_TCGAKICH, slide_nuclearLevel_label_TCGAKICH, slide_prognosis_TCGAKICH = get_sort_one_center(os.path.join(data_root, "output_TCGA_kich_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/TCGA病理.xlsx", center='TCGA')
    print("TCGA_kich Load")

    slide_patch_feat_TCGAKIRC, slide_patch_label_TCGAKIRC, slide_fileName_TCGAKIRC, slide_patch_fileName_TCGAKIRC, slide_patho_label_TCGAKIRC, slide_nuclearLevel_label_TCGAKIRC, slide_prognosis_TCGAKIRC = get_sort_one_center(os.path.join(data_root, "output_TCGA_kirc_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/TCGA病理.xlsx", center='TCGA')
    print("TCGA_kirc Load")

    slide_patch_feat_TCGAKIRP, slide_patch_label_TCGAKIRP, slide_fileName_TCGAKIRP, slide_patch_fileName_TCGAKIRP, slide_patho_label_TCGAKIRP, slide_nuclearLevel_label_TCGAKIRP, slide_prognosis_TCGAKIRP = get_sort_one_center(os.path.join(data_root, "output_TCGA_kirp_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/TCGA病理.xlsx", center='TCGA')
    print("TCGA_kirp Load")

    slide_patch_feat_CCRCC, slide_patch_label_CCRCC, slide_fileName_CCRCC, slide_patch_fileName_CCRCC, slide_patho_label_CCRCC, slide_nuclearLevel_label_CCRCC, slide_prognosis_CCRCC = get_sort_one_center(os.path.join(data_root, "output_CCRCC_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/CPTAC病理.xlsx", center='CCRCC')
    print("CCRCC Load")

    slide_patch_feat_ZSRUTOU, slide_patch_label_ZSRUTOU, slide_fileName_ZSRUTOU, slide_patch_fileName_ZSRUTOU, slide_patho_label_ZSRUTOU, slide_nuclearLevel_label_ZSRUTOU, slide_prognosis_ZSRUTOU = get_sort_one_center(os.path.join(data_root, "output_zhongshan_rutou_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/ZS_临床数据.xlsx", center='ZS')
    print("zhongshan_rutou Load")

    slide_patch_feat_ZSXIANSE, slide_patch_label_ZSXIANSE, slide_fileName_ZSXIANSE, slide_patch_fileName_ZSXIANSE, slide_patho_label_ZSXIANSE, slide_nuclearLevel_label_ZSXIANSE, slide_prognosis_ZSXIANSE = get_sort_one_center(os.path.join(data_root, "output_zhongshan_xianse_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/ZS_临床数据.xlsx", center='ZS')
    print("zhongshan_xianse Load")

    slide_patch_feat_ZSXINTOU, slide_patch_label_ZSXINTOU, slide_fileName_ZSXINTOU, slide_patch_fileName_ZSXINTOU, slide_patho_label_ZSXINTOU, slide_nuclearLevel_label_ZSXINTOU, slide_prognosis_ZSXINTOU = get_sort_one_center(os.path.join(data_root, "output_zhongshan_xintou_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/ZS_临床数据.xlsx", center='ZS')
    print("zhongshan_xintou Load")

    slide_patch_feat_ZSBUCHONG, slide_patch_label_ZSBUCHONG, slide_fileName_ZSBUCHONG, slide_patch_fileName_ZSBUCHONG, slide_patho_label_ZSBUCHONG, slide_nuclearLevel_label_ZSBUCHONG, slide_prognosis_ZSBUCHONG = get_sort_one_center(os.path.join(data_root, "output_zhongshan_buchong_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/ZS_临床数据.xlsx", center='ZS')
    print("zhongshan_buchong Load")

    slide_patch_feat_ZSQITA, slide_patch_label_ZSQITA, slide_fileName_ZSQITA, slide_patch_fileName_ZSQITA, slide_patho_label_ZSQITA, slide_nuclearLevel_label_ZSQITA, slide_prognosis_ZSQITA = get_sort_one_center(os.path.join(data_root, "output_zhongshan_qita_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path="/home/xiaoyuan/Data3/dataset_Renal_Inhouse/ZS_临床数据.xlsx", center='ZS')
    print("zhongshan_qita Load")


    # Internal
    Internal_data = slide_patch_feat_ZSRUTOU + slide_patch_feat_ZSXIANSE + slide_patch_feat_ZSQITA + slide_patch_feat_ZSXINTOU + slide_patch_feat_ZSBUCHONG
    Internal_label = slide_patch_label_ZSRUTOU + slide_patch_label_ZSXIANSE + slide_patch_label_ZSQITA + slide_patch_label_ZSXINTOU + slide_patch_label_ZSBUCHONG
    Internal_name_slide = slide_fileName_ZSRUTOU + slide_fileName_ZSXIANSE + slide_fileName_ZSQITA + slide_fileName_ZSXINTOU + slide_fileName_ZSBUCHONG
    Internal_name_patch = slide_patch_fileName_ZSRUTOU + slide_patch_fileName_ZSXIANSE + slide_patch_fileName_ZSQITA + slide_patch_fileName_ZSXINTOU + slide_patch_fileName_ZSBUCHONG
    Internal_patho_label = slide_patho_label_ZSRUTOU + slide_patho_label_ZSXIANSE + slide_patho_label_ZSQITA + slide_patho_label_ZSXINTOU + slide_patho_label_ZSBUCHONG
    Internal_nuclearLevel_label = slide_nuclearLevel_label_ZSRUTOU + slide_nuclearLevel_label_ZSXIANSE + slide_nuclearLevel_label_ZSQITA + slide_nuclearLevel_label_ZSXINTOU + slide_nuclearLevel_label_ZSBUCHONG
    Internal_prognosis = slide_prognosis_ZSRUTOU + slide_prognosis_ZSXIANSE + slide_prognosis_ZSQITA + slide_prognosis_ZSXINTOU + slide_prognosis_ZSBUCHONG

    # External
    External_data = slide_patch_feat_XIAMEN + slide_patch_feat_ZHANGYE + slide_patch_feat_HUADONG
    External_label = slide_patch_label_XIAMEN + slide_patch_label_ZHANGYE + slide_patch_label_HUADONG
    External_name_slide = slide_fileName_XIAMEN + slide_fileName_ZHANGYE+ slide_fileName_HUADONG
    External_name_patch = slide_patch_fileName_XIAMEN + slide_patch_fileName_ZHANGYE + slide_patch_fileName_HUADONG
    External_patho_label = slide_patho_label_XIAMEN + slide_patho_label_ZHANGYE + slide_patho_label_HUADONG
    External_nuclearLevel_label = slide_nuclearLevel_label_XIAMEN + slide_nuclearLevel_label_ZHANGYE + slide_nuclearLevel_label_HUADONG
    External_prognosis = slide_prognosis_XIAMEN + slide_prognosis_ZHANGYE + slide_prognosis_HUADONG

    # External-web
    ExternalWeb_data = slide_patch_feat_TCGAKICH + slide_patch_feat_TCGAKIRC + slide_patch_feat_TCGAKIRP + slide_patch_feat_CCRCC
    ExternalWeb_label = slide_patch_label_TCGAKICH + slide_patch_label_TCGAKIRC + slide_patch_label_TCGAKIRP + slide_patch_label_CCRCC
    ExternalWeb_name_slide = slide_fileName_TCGAKICH + slide_fileName_TCGAKIRC + slide_fileName_TCGAKIRP + slide_fileName_CCRCC
    ExternalWeb_name_patch = slide_patch_fileName_TCGAKICH + slide_patch_fileName_TCGAKIRC + slide_patch_fileName_TCGAKIRP + slide_patch_fileName_CCRCC
    ExternalWeb_patho_label = slide_patho_label_TCGAKICH + slide_patho_label_TCGAKIRC + slide_patho_label_TCGAKIRP + slide_patho_label_CCRCC
    ExternalWeb_nuclearLevel_label = slide_nuclearLevel_label_TCGAKICH + slide_nuclearLevel_label_TCGAKIRC + slide_nuclearLevel_label_TCGAKIRP + slide_nuclearLevel_label_CCRCC
    ExternalWeb_prognosis = slide_prognosis_TCGAKICH + slide_prognosis_TCGAKIRC + slide_prognosis_TCGAKIRP + slide_prognosis_CCRCC



    # shuffle
    Internal_all = list(zip(Internal_data, Internal_label, Internal_name_slide, Internal_name_patch, Internal_patho_label, Internal_nuclearLevel_label, Internal_prognosis))
    random.shuffle(Internal_all)
    Internal_data[:], Internal_label[:], Internal_name_slide[:], Internal_name_patch[:], Internal_patho_label[:], Internal_nuclearLevel_label[:], Internal_prognosis[:] = zip(*Internal_all)

    # shuffle
    testExternal_all = list(zip(External_data, External_label, External_name_slide, External_name_patch, External_patho_label, External_nuclearLevel_label, External_prognosis))
    random.shuffle(testExternal_all)
    External_data[:], External_label[:], External_name_slide[:], External_name_patch[:], External_patho_label[:], External_nuclearLevel_label[:], External_prognosis[:] = zip(*testExternal_all)

    # shuffle
    testExternalWeb_all = list(zip(ExternalWeb_data, ExternalWeb_label, ExternalWeb_name_slide, ExternalWeb_name_patch, ExternalWeb_patho_label, ExternalWeb_nuclearLevel_label, ExternalWeb_prognosis))
    random.shuffle(testExternalWeb_all)
    ExternalWeb_data[:], ExternalWeb_label[:], ExternalWeb_name_slide[:], ExternalWeb_name_patch[:], ExternalWeb_patho_label[:], ExternalWeb_nuclearLevel_label[:], ExternalWeb_prognosis[:] = zip(*testExternalWeb_all)


    # Further split internal train and internal test
    split_ratio = 0.7
    num_InternalTrain = int(len(Internal_all)*split_ratio)

    InternalTrain_data = Internal_data[:num_InternalTrain]
    InternalTrain_label = Internal_label[:num_InternalTrain]
    InternalTrain_name_slide = Internal_name_slide[:num_InternalTrain]
    InternalTrain_name_patch = Internal_name_patch[:num_InternalTrain]
    InternalTrain_patho_label = Internal_patho_label[:num_InternalTrain]
    InternalTrain_nuclearLevel_label = Internal_nuclearLevel_label[:num_InternalTrain]
    InternalTrain_prognosis_label = Internal_prognosis[:num_InternalTrain]

    InternalTest_data = Internal_data[num_InternalTrain:]
    InternalTest_label = Internal_label[num_InternalTrain:]
    InternalTest_name_slide = Internal_name_slide[num_InternalTrain:]
    InternalTest_name_patch = Internal_name_patch[num_InternalTrain:]
    InternalTest_patho_label = Internal_patho_label[num_InternalTrain:]
    InternalTest_nuclearLevel_label = Internal_nuclearLevel_label[num_InternalTrain:]
    InternalTest_prognosis_label = Internal_prognosis[num_InternalTrain:]

    # Split some external test into train
    split_ratio_external = 0.0
    num_splitExternal = int(len(testExternal_all)*split_ratio_external)
    ExternalTest_data = External_data[num_splitExternal:]
    ExternalTest_label = External_label[num_splitExternal:]
    ExternalTest_name_slide = External_name_slide[num_splitExternal:]
    ExternalTest_name_patch = External_name_patch[num_splitExternal:]
    ExternalTest_patho_label = External_patho_label[num_splitExternal:]
    ExternalTest_nuclearLevel_label = External_nuclearLevel_label[num_splitExternal:]
    ExternalTest_prognosis_label = External_prognosis[num_splitExternal:]

    # InternalTrain_data = InternalTrain_data + test_data[:num_splitExternal]
    # InternalTrain_label = InternalTrain_label + test_label[:num_splitExternal]
    # InternalTrain_name_slide = InternalTrain_name_slide + test_name_slide[:num_splitExternal]
    # InternalTrain_name_patch = InternalTrain_name_patch + test_name_patch[:num_splitExternal]

    print("ALL FEAT LOADED")
    return (
        [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label],
        [InternalTest_data, InternalTest_label, InternalTest_name_slide, InternalTest_name_patch, InternalTest_patho_label, InternalTest_nuclearLevel_label, InternalTest_prognosis_label],
        [ExternalTest_data, ExternalTest_label, ExternalTest_name_slide, ExternalTest_name_patch, ExternalTest_patho_label, ExternalTest_nuclearLevel_label, ExternalTest_prognosis_label]
    )


def get_train_test_ds_OnlyTCGA_region(
        data_root='/home/science/Downloads/nips2024data/d69acdab5e934806b3af57120b593535/tcga',
        downsample=1.0):

    slide_patch_feat_TCGAKICH, slide_patch_label_TCGAKICH, slide_fileName_TCGAKICH, slide_patch_fileName_TCGAKICH, slide_patho_label_TCGAKICH, slide_nuclearLevel_label_TCGAKICH, slide_prognosis_TCGAKICH = get_sort_one_center(os.path.join(data_root, "feat/output_TCGA_kich_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path=os.path.join(data_root, "TCGA病理.xlsx"), center='TCGA')
    print("TCGA_kich Load")

    slide_patch_feat_TCGAKIRC, slide_patch_label_TCGAKIRC, slide_fileName_TCGAKIRC, slide_patch_fileName_TCGAKIRC, slide_patho_label_TCGAKIRC, slide_nuclearLevel_label_TCGAKIRC, slide_prognosis_TCGAKIRC = get_sort_one_center(os.path.join(data_root, "feat/output_TCGA_kirc_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path=os.path.join(data_root, "TCGA病理.xlsx"), center='TCGA')
    print("TCGA_kirc Load")

    slide_patch_feat_TCGAKIRP, slide_patch_label_TCGAKIRP, slide_fileName_TCGAKIRP, slide_patch_fileName_TCGAKIRP, slide_patho_label_TCGAKIRP, slide_nuclearLevel_label_TCGAKIRP, slide_prognosis_TCGAKIRP = get_sort_one_center(os.path.join(data_root, "feat/output_TCGA_kirp_feat_224x224_mpp_1_0_CLIP(ViTB32)"), downsample=downsample, slide_patho_anno_path=os.path.join(data_root, "TCGA病理.xlsx"), center='TCGA')
    print("TCGA_kirp Load")

    # External-web
    ExternalWeb_data = slide_patch_feat_TCGAKICH + slide_patch_feat_TCGAKIRC + slide_patch_feat_TCGAKIRP
    ExternalWeb_label = slide_patch_label_TCGAKICH + slide_patch_label_TCGAKIRC + slide_patch_label_TCGAKIRP
    ExternalWeb_name_slide = slide_fileName_TCGAKICH + slide_fileName_TCGAKIRC + slide_fileName_TCGAKIRP
    ExternalWeb_name_patch = slide_patch_fileName_TCGAKICH + slide_patch_fileName_TCGAKIRC + slide_patch_fileName_TCGAKIRP
    ExternalWeb_patho_label = slide_patho_label_TCGAKICH + slide_patho_label_TCGAKIRC + slide_patho_label_TCGAKIRP
    ExternalWeb_nuclearLevel_label = slide_nuclearLevel_label_TCGAKICH + slide_nuclearLevel_label_TCGAKIRC + slide_nuclearLevel_label_TCGAKIRP
    ExternalWeb_prognosis = slide_prognosis_TCGAKICH + slide_prognosis_TCGAKIRC + slide_prognosis_TCGAKIRP

    # shuffle
    testExternalWeb_all = list(zip(ExternalWeb_data, ExternalWeb_label, ExternalWeb_name_slide, ExternalWeb_name_patch, ExternalWeb_patho_label, ExternalWeb_nuclearLevel_label, ExternalWeb_prognosis))
    random.shuffle(testExternalWeb_all)
    ExternalWeb_data[:], ExternalWeb_label[:], ExternalWeb_name_slide[:], ExternalWeb_name_patch[:], ExternalWeb_patho_label[:], ExternalWeb_nuclearLevel_label[:], ExternalWeb_prognosis[:] = zip(*testExternalWeb_all)

    # Further split internal train and internal test
    split_ratio = 0.7
    num_InternalTrain = int(len(testExternalWeb_all)*split_ratio)

    InternalTrain_data = ExternalWeb_data[:num_InternalTrain]
    InternalTrain_label = ExternalWeb_label[:num_InternalTrain]
    InternalTrain_name_slide = ExternalWeb_name_slide[:num_InternalTrain]
    InternalTrain_name_patch = ExternalWeb_name_patch[:num_InternalTrain]
    InternalTrain_patho_label = ExternalWeb_patho_label[:num_InternalTrain]
    InternalTrain_nuclearLevel_label = ExternalWeb_nuclearLevel_label[:num_InternalTrain]
    InternalTrain_prognosis_label = ExternalWeb_prognosis[:num_InternalTrain]

    InternalTest_data = ExternalWeb_data[num_InternalTrain:]
    InternalTest_label = ExternalWeb_label[num_InternalTrain:]
    InternalTest_name_slide = ExternalWeb_name_slide[num_InternalTrain:]
    InternalTest_name_patch = ExternalWeb_name_patch[num_InternalTrain:]
    InternalTest_patho_label = ExternalWeb_patho_label[num_InternalTrain:]
    InternalTest_nuclearLevel_label = ExternalWeb_nuclearLevel_label[num_InternalTrain:]
    InternalTest_prognosis_label = ExternalWeb_prognosis[num_InternalTrain:]

    print("ALL FEAT LOADED")
    return (
        [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label],
        [InternalTest_data, InternalTest_label, InternalTest_name_slide, InternalTest_name_patch, InternalTest_patho_label, InternalTest_nuclearLevel_label, InternalTest_prognosis_label],
    )


class Region_5Classes_Feat(torch.utils.data.Dataset):
    def __init__(self, ds_data, ds_label, return_bag=False):
        self.all_slide_feat = ds_data
        self.all_slide_label = ds_label
        self.return_bag = return_bag

        if not return_bag:
            self.all_patches_feat = np.concatenate(self.all_slide_feat)
            self.all_patches_label = np.concatenate(self.all_slide_label)
            print('============= Instance Labels =============')
            statistics_dataset(self.all_patches_label)

    def __getitem__(self, index):
        if self.return_bag:
            img = self.all_slide_feat[index]
            label = self.all_slide_label[index]
            return img, label, index
        else:
            img = self.all_patches_feat[index]
            label = self.all_patches_label[index]
            return img, label, index

    def __len__(self):
        if self.return_bag:
            return len(self.all_slide_feat)
        else:
            return len(self.all_patches_feat)


def translate_patho_label(slide_i_raw_patho_label):
    if slide_i_raw_patho_label == 0:
        slide_i_patho_label = 1
    elif slide_i_raw_patho_label == 2:
        slide_i_patho_label = 2
    elif slide_i_raw_patho_label == 6:
        slide_i_patho_label = 3
    else:
        raise
    return slide_i_patho_label


class TumorRegion_PathologyType_Feat(torch.utils.data.Dataset):
    def __init__(self, ds_data_all):
        self.all_raw_slide_feat = ds_data_all[0]
        self.all_slide_patch_label = ds_data_all[1]
        self.all_slide_name = ds_data_all[2]
        self.all_slide_patho_label = ds_data_all[4]

        self.all_slide_feat = []
        self.slide_label_all = []
        self.slide_patch_label_all = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        slide_index = 0
        for i in range(len(self.all_raw_slide_feat)):
            num_patch = len(self.all_raw_slide_feat[i])
            slide_i_patho_label = translate_patho_label(self.all_slide_patho_label[i])

            slide_i_patch_patho_label = self.all_slide_patch_label[i]
            slide_i_patch_patho_label[np.where(slide_i_patch_patho_label != 1)] = 0
            slide_i_patch_patho_label[np.where(slide_i_patch_patho_label == 1)] = slide_i_patho_label

            if len(np.where(slide_i_patch_patho_label != 0)[0]) == 0:
                print("[DATA] skip slide without Tumor Region: {}".format(self.all_slide_name[i]))
                continue

            self.all_slide_feat.append(self.all_raw_slide_feat[i])
            self.slide_label_all.append(slide_i_patho_label)
            self.slide_patch_label_all.append(slide_i_patch_patho_label)
            self.patch_corresponding_slide_label.append(np.repeat(slide_i_patho_label, num_patch))
            self.patch_corresponding_slide_index.append(np.repeat(slide_index, num_patch))
            slide_index = slide_index + 1
        self.patch_corresponding_slide_label = np.concatenate(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.concatenate(self.patch_corresponding_slide_index)

        print('============= Bag Pathology Labels =============')
        statistics_dataset(self.slide_label_all)
        print('============= Patch Pathology Labels =============')
        statistics_dataset(np.concatenate(self.all_slide_patch_label))
        print('============= Slide Pos Patch Ratio =============')
        all_slide_pos_ratio = []
        for i in range(len(self.all_slide_feat)):
            num_patch = self.all_slide_feat[i].shape[0]
            all_slide_pos_ratio.append(len(np.where(self.slide_patch_label_all[i]!=0)[0])/num_patch)
        all_slide_pos_ratio = np.array(all_slide_pos_ratio)
        print(np.histogram(all_slide_pos_ratio))

    def __getitem__(self, index):
        slide_feat = self.all_slide_feat[index]
        slide_label = self.slide_label_all[index]
        slide_patch_label = self.slide_patch_label_all[index]
        return slide_feat, [slide_patch_label, slide_label, index], index

    def __len__(self):
        return len(self.all_slide_feat)


if __name__ == '__main__':
    InternalTrain, InternalTest = get_train_test_ds_OnlyTCGA_region(downsample=1.0)
    # InternalTrain, InternalTest, ExternalTest = get_train_test_ds_MultiCenter_region(downsample=0.3)
    # train_ds = TumorRegion_PathologyType_Feat(InternalTrain)
    # test_ds =TumorRegion_PathologyType_Feat(InternalTest)
    train_ds = TumorRegion_PathologyType_Feat(InternalTrain)
    test_ds =TumorRegion_PathologyType_Feat(InternalTest)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    for img, label, _ in train_loader:
        print(img.shape)
        print(label[0])
    print("END")

