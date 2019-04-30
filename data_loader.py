import scipy
from glob import glob
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name, img_res = (128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size = 1, upscale = 4, is_testing = False):
        data_type = "train" if not is_testing else "test"
        
        path = glob('%s/*' % (self.dataset_name))
        
        if(batch_size == 0):
            batch_images = np.random.choice(path, size = len(path))
        else:
            batch_images = np.random.choice(path, size = batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h / upscale), int(w / upscale)

            img_hr = misc.imresize(img, self.img_res, interp = 'bicubic')
            # img_gauss = ndimage.gaussian_filter(img_hr, 5)
            img_lr = misc.imresize(img_hr, (low_h, low_w), interp = 'bicubic')

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


    def imread(self, path):
        return misc.imread(path, mode = 'RGB').astype(np.float)
