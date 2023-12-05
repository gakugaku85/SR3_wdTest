from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import SimpleITK as sitk
import data.util as Util
from tqdm import tqdm


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.hr_imgs = []
        self.sr_imgs = []

        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'mhd':
            self.sr_path = Util.get_paths_from_mhds(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_mhds(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_mhds(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

        for hr, sr in tqdm(zip(self.hr_path, self.sr_path), desc='create datasets', total=len(self.hr_path)):
            img_HR = sitk.ReadImage(hr)
            img_SR = sitk.ReadImage(sr)
            nda_img_HR = sitk.GetArrayFromImage(img_HR)
            nda_img_SR = sitk.GetArrayFromImage(img_SR)
            self.hr_imgs.append(nda_img_HR)
            self.sr_imgs.append(nda_img_SR)
        print("train_slice_length : {}".format(self.data_len))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        nda_img_HR = self.hr_imgs[index]
        nda_img_SR = self.sr_imgs[index]

        if self.datatype == 'mhd':
            img_HR = Image.fromarray(nda_img_HR)
            img_SR = Image.fromarray(nda_img_SR)
            [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(0, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
