import collections
import torch
import numpy as np
import os
from PIL import Image

from torch.utils import data


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class WASRGBDLoader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=False,
        img_size=(80, 80),
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = []
        self.depth_files = []
        
        data_subfolder = [x[1] for x in os.walk(root)]
        self.class_split = data_subfolder[0]
        self.split = self.class_split
        self.n_classes = len(self.class_split)
        self.lable_finder = [] #

        class_name = 0
        for split in self.class_split:
            if(self.test_mode):
              print(self.class_split)
              print(split)
            
            file_list = sorted(recursive_glob(rootdir=self.root + split + "/", suffix="_crop.png"))
            self.files.extend(file_list)
            self.lable_finder.extend([class_name]*len(file_list))
            class_name = class_name+1

        for split in self.class_split:
            file_list = sorted(
                recursive_glob(rootdir=self.root + split + "/", suffix="_depthcrop.png")
            )
            self.depth_files.extend(file_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()
        dpt_path = self.depth_files[index].rstrip()
        # img_number = img_path.split('/')[-1]
        # dpt_path = os.path.join(self.root, 'annotations', img_number).replace('jpg', 'png')

        with open(img_path, 'rb') as f:
          img = Image.open(f)
          img = img.convert('RGB')
          img = img.resize((self.img_size[0], self.img_size[1]))
        img = np.array(img, dtype=np.uint8)

        with open(dpt_path, 'rb') as f:
          dpt = Image.open(f)
          dpt = dpt.convert('RGB')
          dpt = dpt.resize((self.img_size[0], self.img_size[1]),Image.NEAREST)
        dpt = np.array(dpt, dtype=np.uint8)
        dpt = dpt[1]

        #if not (len(img.shape) == 3 and len(dpt.shape) == 2):
        #    print("img",len(img.shape),"dpt",len(dpt.shape))
        #    return self.__getitem__(np.random.randint(0, self.__len__()))


        # if self.augmentations is not None:
        #     img = self.augmentations(img)
        #     #dpt = self.augmentations(dpt)
        #     #img, dpt = self.augmentations(img, dpt)
            
        # if self.is_transform:
        #     img, dpt = self.transform(img, dpt)

        return img, self.lable_finder[index]


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    local_path = "/home/meet/datasets/SUNRGBD/"
    dst = WASRGBDLoader(local_path)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()