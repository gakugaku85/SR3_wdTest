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
        self.frangi_imgs = []

        if datatype == 'mhd':
            self.sr_path = Util.get_paths_from_mhds(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_mhds(
                '{}/hr_{}'.format(dataroot, r_resolution))
            self.frangi_path = Util.get_paths_from_mhds(
                '{}/frangi'.format(dataroot))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

        for hr, sr, frangi in tqdm(zip(self.hr_path, self.sr_path, self.frangi_path), desc='create datasets', total=len(self.hr_path)):
            img_HR = sitk.ReadImage(hr)
            img_SR = sitk.ReadImage(sr)
            img_frangi = sitk.ReadImage(frangi)
            nda_img_HR = sitk.GetArrayFromImage(img_HR)
            nda_img_SR = sitk.GetArrayFromImage(img_SR)
            nda_img_frangi = sitk.GetArrayFromImage(img_frangi)
            self.hr_imgs.append(nda_img_HR)
            self.sr_imgs.append(nda_img_SR)
            self.frangi_imgs.append(nda_img_frangi)
        print("train_slice_length : {}".format(self.data_len))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None

        nda_img_HR = self.hr_imgs[index]
        nda_img_SR = self.sr_imgs[index]
        nda_img_frangi = self.frangi_imgs[index]

        if self.datatype == 'mhd':
            img_HR = Image.fromarray(nda_img_HR)
            img_SR = Image.fromarray(nda_img_SR)
            frangi_img = Image.fromarray(nda_img_frangi)
            [img_SR, img_HR, frangi_img] = Util.transform_augment([img_SR, img_HR, frangi_img], split=self.split, min_max=(0, 1))
            return {'HR': img_HR, 'SR': img_SR, "frangi": frangi_img ,'Index': index}
