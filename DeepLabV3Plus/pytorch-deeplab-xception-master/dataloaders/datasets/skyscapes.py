#skyscapes2.py
import os
import numpy as np
from PIL import Image
from mypath import Path
from torchvision import transforms
import torch.utils.data as data
import torch

SEG_LABELS_LIST = [
    {"id": 0, "name": "Background",                     "rgb_values": [  0,    0,    0]},    # Non-lane-marking.
    {"id": 1,  "name": "Dash Line",                     "rgb_values": [255,    0,    0]},    # Any broken line with long line segments, e.g., lane separators.
    {"id": 2,  "name": "Long Line",                     "rgb_values": [  0,    0,  255]},    # Thin solid lines, such as no passing lines or roadside markings.
    {"id": 3,  "name": "Small dash line",               "rgb_values": [255,  255,    0]},    # Any broken line with tiny line segments, e.g., lines enclosing pedestrian crossings.
    {"id": 4,  "name": "turn signs",                    "rgb_values": [  0,  255,    0]},    # Arrows on the road, such as intersection arrows or merge arrows.
    {"id": 5,  "name": "other signs",                   "rgb_values": [255,  128,    0]},    # All other signs, e.g., numbers that indicate the speed limit.
    {"id": 6,  "name": "Plus sign on crossroads",       "rgb_values": [128,    0,    0]},    # All crossing tiny lines.
    {"id": 7,  "name": "Srosswalk",                     "rgb_values": [  0,  255,  255]},    # Zebra-striped markings across the roadway mark a pedestrian crosswalk.
    {"id": 8,  "name": "Stop line",                     "rgb_values": [  0,  128,    0]},    # Thick solid line across lanes that signal to stop behind the line.
    {"id": 9,  "name": "Zebra zone",                    "rgb_values": [255,    0,  255]},    # Areas with diagonal lines, e.g., restricted zones.
    {"id": 10, "name": "No parking zone",               "rgb_values": [  0,  150,  150]},    # Zig-zag lines next to the curb mark that indicate that stopping or parking is forbidden.
    {"id": 11, "name": "Parking space",                 "rgb_values": [200,  200,    0]},    # Any line that marks parking spots.
    {"id": 12, "name": "Other lane-markings",           "rgb_values": [100,    0,  200]},    # All other lane-markings.

]

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


class SkyscapesDataset(data.Dataset):
    NUM_CLASSES = 13

    def __init__(self, split="train"):

        self.root = Path.db_root_dir('skyscapes')
        self.split = split
        #self.args = args
        self.files = {}
       
        # Set up image and mask paths
        self.images_base = os.path.join(self.root,self.split,"images")
        self.annotations_base = os.path.join(self.root, self.split, "labels")

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

        img = Image.open(os.path.join(self.root,self.split,
                                      'images',
                                      img_id + '.png')).convert('RGB')
        #center_crop = transforms.CenterCrop(240)
        #img = center_crop(img)
        img = to_tensor(img)
        img = norm_tensor(img)

        target = Image.open(os.path.join(self.root,self.split,
                                         'labels',
                                         img_id + '_mask.png'))
        #target = center_crop(target)
        target = np.array(target, dtype=np.int64)

        target_labels = target[..., 0]
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())

        return img, target_labels
    