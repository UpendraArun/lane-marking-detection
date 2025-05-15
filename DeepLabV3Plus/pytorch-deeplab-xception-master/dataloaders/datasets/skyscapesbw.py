#skyscapesbw.py
import os
import numpy as np
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
import torch.utils.data as data
import torch

SEG_LABELS_LIST = [
    {"id": 0, "name": "Background",                     "rgb_values": [  0,    0,    0]},    # Non-lane-marking.
    {"id": 1,  "name": "Dash Line",                     "rgb_values": [255,    255,    255]},    # Any lane marking.
]


def label_img_to_rgb(label_img):
    label_img_rgb = np.stack([label_img, label_img, label_img], axis=-1)
    return label_img_rgb.astype(np.uint8)

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


class SkyscapesDataset(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, split="train"):

        self.root = Path.db_root_dir('skyscapes')
        self.split = split
        #self.args = args
        self.files = {}
       
        # Set up image and mask paths
        self.images_base = os.path.join(self.root,self.split,"images")
        self.annotations_base = os.path.join(self.root, self.split, "grayscale")

        self.image_names = [filename.split('.')[0] for filename in os.listdir(self.images_base) if filename.endswith('.png')]
    

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")
    
    
    def __len__(self):
        return len(self.image_names)
    

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        norm_tensor = transforms.Normalize(RGB_MEAN, RGB_STD)
        img_id = self.image_names[index].replace('.png', '')

        img = Image.open(os.path.join(self.root, self.split, 'images', img_id + '.png')).convert('RGB')
        img = to_tensor(img)
        img = norm_tensor(img)

        target = Image.open(os.path.join(self.root, self.split, 'grayscale', img_id + '_mask.png'))
        target = np.array(target, dtype=np.int64)
        
        # Create a binary label tensor with values 0 and 1
        # Background (0) is where target is 0, Dash Line (1) is where target is not 0
        target_labels = torch.where(torch.tensor(target) == 0, torch.tensor(0), torch.tensor(1))


        return img, target_labels