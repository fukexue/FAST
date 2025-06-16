import numpy as np
import torch
import clip
from Datasets_loader.dataset_CAMELYON16_new import CAMELYON_16_10x, CAMELYON_16_5x
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import time
from sklearn.svm import LinearSVC, SVC
import torch.nn as nn
import torch.nn.functional as F
import utliz
import h5py


def compute_CLIP_feat(model, dataloader):
    L = len(dataloader.dataset)
    feat_all = np.zeros([L, 1024], dtype=np.float32)
    for data in tqdm(dataloader, desc='Computing features'):
        images = data[0].cuda()
        selected = data[2]
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            feat_all[selected, :] = image_features.detach().cpu().numpy()
    return feat_all


def save_CLIP_feat():
    model, preprocess = clip.load("RN50")

    train_ds = CAMELYON_16_5x(train=True, transform=preprocess, downsample=1.0, return_bag=False)
    val_ds = CAMELYON_16_5x(train=False, transform=preprocess, downsample=1.0, return_bag=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    output_dir = "./output_CAMELYON_feat_224x224_5x_CLIP(RN50)"

    feat_train = compute_CLIP_feat(model, train_loader)
    h5f = h5py.File(os.path.join(output_dir, 'train_patch_feat.h5'), 'w')
    h5f.create_dataset('dataset_1', data=feat_train)
    np.save(os.path.join(output_dir, "train_patch_label.npy"), train_loader.dataset.patch_label)
    np.save(os.path.join(output_dir, "train_patch_corresponding_slide_label.npy"), train_loader.dataset.patch_corresponding_slide_label)
    np.save(os.path.join(output_dir, "train_patch_corresponding_slide_index.npy"), train_loader.dataset.patch_corresponding_slide_index)
    np.save(os.path.join(output_dir, "train_patch_corresponding_slide_name.npy"), train_loader.dataset.patch_corresponding_slide_name)

    feat_val = compute_CLIP_feat(model, val_loader)
    h5f = h5py.File(os.path.join(output_dir, 'val_patch_feat.h5'), 'w')
    h5f.create_dataset('dataset_1', data=feat_val)
    np.save(os.path.join(output_dir, "val_patch_label.npy"), val_loader.dataset.patch_label)
    np.save(os.path.join(output_dir, "val_patch_corresponding_slide_label.npy"), val_loader.dataset.patch_corresponding_slide_label)
    np.save(os.path.join(output_dir, "val_patch_corresponding_slide_index.npy"), val_loader.dataset.patch_corresponding_slide_index)
    np.save(os.path.join(output_dir, "val_patch_corresponding_slide_name.npy"), val_loader.dataset.patch_corresponding_slide_name)
    print("END")


def eval_CLIP_feat():
    # Load saved CLIP feat
    feat_dir = "./output_CAMELYON_feat_224x224_5x_CLIP(RN50)"
    feat_train = h5py.File((os.path.join(feat_dir, "train_patch_feat.h5")), 'r')['dataset_1'][:]
    feat_val = h5py.File((os.path.join(feat_dir, "val_patch_feat.h5")), 'r')['dataset_1'][:]
    label_train = np.load(os.path.join(feat_dir, "train_patch_label.npy"))
    label_val = np.load(os.path.join(feat_dir, "val_patch_label.npy"))
    print("Feat Loaded")

    test_acc, test_auc = evaluate(evaluateMethod='SVM',
                                  train_feat=feat_train, train_label=label_train,
                                  val_feat=feat_val, val_label=label_val, FC_eval_epochs=200,
                                  device='cuda:0')
    return test_acc, test_auc


def evaluate(evaluateMethod, train_feat, train_label, val_feat, val_label, device, FC_eval_epochs,
             patch_from_neg_slide_train_index=None):
    t0 = time.time()
    if evaluateMethod == 'FC':
        train_feat = torch.from_numpy(train_feat).to(device)
        train_label = torch.from_numpy(train_label).to(device)
        val_feat = torch.from_numpy(val_feat).to(device)
        val_label = torch.from_numpy(val_label).to(device)
        # classifier = nn.Linear(2048, 2).to(device)
        classifier = nn.Linear(512, 2).to(device)
        optim = torch.optim.Adam(classifier.parameters(), lr=1e-3, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.2, milestones=[60,80])
        batch_size = 128
        for epoch in tqdm(range(FC_eval_epochs), desc='FC training'):
            for i in range(train_feat.shape[0] // batch_size):
                optim.zero_grad()
                feats = train_feat[i*batch_size: (i+1)*batch_size, :]
                labels = train_label[i*batch_size: (i+1)*batch_size, :]
                logits = classifier(feats)
                loss = F.cross_entropy(logits, labels.to(device).squeeze())
                loss.backward()
                optim.step()
            scheduler.step()
        # only evaluate encoder after classifier finish training in train_loader
        correct = 0
        val_prob_all = []
        with torch.no_grad():
            for i in range(val_feat.shape[0] // batch_size):
                feats = val_feat[i*batch_size: (i+1)*batch_size, :]
                labels = val_label[i*batch_size: (i+1)*batch_size, :]
                outputs = classifier(feats)
                pred_prob = F.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1)
                correct += (pred.cpu() == labels.squeeze().cpu()).sum().item()
                val_prob_all.append(pred_prob.detach().cpu().numpy())
        val_acc = correct / val_feat.shape[0]
        val_prob_all = np.concatenate(val_prob_all, axis=0)
        val_auc = utliz.cal_auc(val_label[:val_prob_all.shape[0]], val_prob_all[:, 1])
        print(f"Epoch {epoch}/{FC_eval_epochs} FC_val_acc {val_acc * 100:.4g}%\tFC_val_AUC {val_auc * 100:.4g}\tcost {time.time()-t0}s")

    elif evaluateMethod == 'SVM':
        t0 = time.time()
        clf = SVC(kernel='linear', probability=True, random_state=0)
        # clf = LinearSVC()
        clf.fit(train_feat, train_label.squeeze())
        pred = clf.predict(val_feat)
        pred_prob = clf.decision_function(val_feat)
        val_acc = np.sum(val_label.squeeze() == pred) * 1. / pred.shape[0]
        val_auc = utliz.cal_auc(val_label, pred_prob)
        print(f"SVM_val_acc:{val_acc * 100:.4g}% SVM_val_AUC:{val_auc :.4g} cost {time.time()-t0}s")

    elif evaluateMethod == 'CLUSTER':
        t0 = time.time()
        neg_center = np.mean(train_feat[patch_from_neg_slide_train_index.nonzero()[0], :], axis=0)
        dist_to_neg_center = np.linalg.norm(val_feat - neg_center[None, :], axis=1)  # L2 distance to neg_center
        pred_prob = dist_to_neg_center/dist_to_neg_center.max()
        val_auc = utliz.cal_auc(val_label, pred_prob)
        val_acc = -1
        print(f"cluster {val_auc :.4g} cost {time.time()-t0}s")

    else:
        val_acc = -1
        val_auc = -1
        print("[ERROR key]")

    return val_acc, val_auc


if __name__ == '__main__':
    # save_CLIP_feat()
    eval_CLIP_feat()