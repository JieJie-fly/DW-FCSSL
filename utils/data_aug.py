from torchvision import transforms
import torch
import numpy as np
import random
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pywt

class wt_mix(object):
    def __init__(self, rate, dim):
        self.rate = rate
        self.dim = dim
        self.wavelet = 'db2'
        self.level = 1

    def __call__(self, x):
        x = torch.tensor(x)
        x_length = len(x)
        cof0 = []
        cof1 = []
        # cof2 = []
        recon_signal = []

        for i in range(x.shape[1]):
            coeffs1 = pywt.wavedec(x[:, i], self.wavelet, level=self.level)
            cof0.append(coeffs1[0])
            cof1.append(coeffs1[1])
            # cof2.append(coeffs1[2])

        for i in range(x.shape[1]):
            coeffs = []
            coeffs.append(cof0[i])
            coeffs.append(random.choice(cof1))
            # coeffs.append(cof1[i])
            # coeffs.append(random.choice(cof2))
            reconstructed_signal = pywt.waverec(coeffs, self.wavelet)
            recon_signal.append(reconstructed_signal)
        recon_signal = np.array(recon_signal).T
        recon_signal = recon_signal[:x_length]
        return recon_signal

class left_cropping(object):
    def __init__(self, left_crop):
        self.left_crop = left_crop
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, x):
        x = x[self.left_crop:]
        return x

class right_cropping(object):
    def __init__(self, right_crop):
        self.right_crop = right_crop
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, x):
        x = x[:-self.right_crop]
        return x

class mask(object):
    def __init__(self, prob):

        self.p = prob
        self.r = 0
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, X):
        B, T, C = X.shape
        # print(B,T,C)
        mask = np.empty_like(X)

        for b in range(B):
            ts = X[b, :, 0]
            ts = np.array(ts)
            et_num = ts[~np.isnan(ts)].size - self.r
            total, num = et_num * C, round(et_num * C * self.p)

            while True:
                i_mask = np.zeros(total)
                i_mask[random.sample(range(total), num)] = 1
                i_mask = i_mask.reshape(et_num, C)
                if 1 not in i_mask.sum(axis=0) and 0 not in i_mask.sum(axis=0):
                    break
                break

            i_mask = np.concatenate((i_mask, np.ones((self.r, C))), axis=0)
            mask[b, ~np.isnan(ts), :] = i_mask
            mask[b, np.isnan(ts), :] = np.nan

        X = np.array(X)
        X[mask == 0] = 0
        return X


class Aug():
    def __init__(self, data_dir, rate, dim, mask_prob, left_crop, right_crop):
        self.files = list(Path(data_dir).rglob("*.csv"))
        self.rate = rate
        self.dim = dim
        self.prob = mask_prob
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.freq_transform = self.get_simclr_transform()[0]
        self.crop_transform = self.get_simclr_transform()[1]
        self.mask_freq_transform = self.get_simclr_transform()[2]
        self.mask_crop_transform = self.get_simclr_transform()[3]
        self.raw_transform = self.get_simclr_transform()[4]

    def get_simclr_transform(self):

        freq_transforms = transforms.Compose([left_cropping(self.left_crop), \
                                             wt_mix(self.rate, self.dim), transforms.ToTensor()])

        crop_transforms = transforms.Compose([right_cropping(self.right_crop), transforms.ToTensor()])

        mask_freq_transforms = transforms.Compose([freq_transforms, mask(self.prob),\
                                                  transforms.ToTensor()])

        mask_crop_transforms = transforms.Compose([crop_transforms, mask(self.prob), \
                                                   transforms.ToTensor()])

        raw_transoforms = transforms.Compose([transforms.ToTensor()])

        return (freq_transforms, crop_transforms, mask_freq_transforms,\
                mask_crop_transforms, raw_transoforms)

    def __getitem__(self, i):
        file_i = str(self.files[i])
        data = pd.read_csv(file_i)
        data = np.array(data)[:, 1:]

        tensor_freq = self.freq_transform(data)
        tensor_crop = self.crop_transform(data)

        # 对增强数据进行掩码处理
        tensor_mask_freq = mask(0.2)(tensor_freq)
        tensor_mask_crop = mask(0.2)(tensor_crop)
        tensor_raw = self.raw_transform(data)

        return tensor_freq, tensor_crop, tensor_mask_freq, tensor_mask_crop, tensor_raw



class CustomDataset(Dataset):
    def __init__(self, freq, crop, mask_freq, mask_crop):
        self.freq = freq
        self.crop = crop
        self.mask_freq = mask_freq
        self.mask_crop = mask_crop
    def __len__(self):
        return len(self.freq)

    def __getitem__(self, idx):
        return self.freq[idx], self.crop[idx], self.mask_freq[idx], self.mask_crop[idx]


