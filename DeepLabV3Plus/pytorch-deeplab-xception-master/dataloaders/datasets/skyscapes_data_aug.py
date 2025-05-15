from PIL import Image
import os
from torch.utils.data import Dataset

class SkyscapesDataset(Dataset):
    NUM_CLASSES = 13  # 12 + 1 background

    def __init__(self, image_dir, grayscale_mask_dir, rgb_mask_dir, transform=None, transform_grayscale_mask=None, transform_rgb_mask=None):
        self.image_dir = image_dir
        self.grayscale_mask_dir = grayscale_mask_dir
        self.rgb_mask_dir = rgb_mask_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
        self.transform_grayscale_mask = transform_grayscale_mask
        self.transform_rgb_mask = transform_rgb_mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        grayscale_mask_name = os.path.join(self.grayscale_mask_dir, self.image_files[idx].replace(".jpg", ".png"))
        rgb_mask_name = os.path.join(self.rgb_mask_dir, self.image_files[idx].replace(".jpg", ".png"))

        image = Image.open(img_name)
        grayscale_mask = Image.open(grayscale_mask_name)
        rgb_mask = Image.open(rgb_mask_name)

        if self.transform:
            image_crops, grayscale_mask_crops, rgb_mask_crops = self.transform(image, grayscale_mask, rgb_mask, self.image_files[idx])

        return image_crops, grayscale_mask_crops, rgb_mask_crops


class SlidingWindowCrop(object):
    def __init__(self, window_size, image_crop_save_dir, grayscale_mask_crop_save_dir, rgb_mask_crop_save_dir, overlap=0.1):
        self.window_size = window_size
        self.image_crop_save_dir = image_crop_save_dir
        self.grayscale_mask_crop_save_dir = grayscale_mask_crop_save_dir
        self.rgb_mask_crop_save_dir = rgb_mask_crop_save_dir
        self.overlap = overlap

    def __call__(self, image, grayscale_mask, rgb_mask, filename):
        image_width, image_height = image.size
        window_width, window_height = self.window_size

        stride_x = int(window_width * (1 - self.overlap))
        stride_y = int(window_height * (1 - self.overlap))

        image_crops = []
        grayscale_mask_crops = []
        rgb_mask_crops = []

        for x in range(0, image_width - window_width + 1, stride_x):
            for y in range(0, image_height - window_height + 1, stride_y):
                box = (x, y, x + window_width, y + window_height)
                image_crop = image.crop(box)
                #print("image_crop:", image_crop)
                grayscale_mask_crop = grayscale_mask.crop(box)
                rgb_mask_crop = rgb_mask.crop(box)

                # Save the crop to image directory
                overlap_percentage = int(self.overlap * 100)
                
                filename_without_extension = os.path.splitext(filename)[0]
                crop_filename_image = os.path.join(self.image_crop_save_dir, f"{filename_without_extension}_crop_{overlap_percentage}_{x}_{y}.jpg")
                #print("Crop_filename_image:", crop_filename_image)
                image_crop.save(crop_filename_image)

                # Save the crop to grayscale mask directory
                crop_filename_grayscale_mask = os.path.join(self.grayscale_mask_crop_save_dir, f"{filename_without_extension}_crop_{overlap_percentage}_{x}_{y}_mask.png")
                grayscale_mask_crop.save(crop_filename_grayscale_mask)

                # Save the crop to RGB mask directory
                crop_filename_rgb_mask = os.path.join(self.rgb_mask_crop_save_dir, f"{filename_without_extension}_crop_{overlap_percentage}_{x}_{y}_mask.png")
                rgb_mask_crop.save(crop_filename_rgb_mask)

                # Append the crops to the list
                image_crops.append(crop_filename_image)
                grayscale_mask_crops.append(crop_filename_grayscale_mask)
                rgb_mask_crops.append(crop_filename_rgb_mask)

                # Apply vertical flip and save
                image_crop_vertical = image_crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                grayscale_mask_crop_vertical = grayscale_mask_crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                rgb_mask_crop_vertical = rgb_mask_crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                crop_filename_image_vertical = crop_filename_image.replace(".jpg", "_vertical.jpg")
                crop_filename_grayscale_mask_vertical = crop_filename_grayscale_mask.replace("_mask.png", "_vertical_mask.png")
                crop_filename_rgb_mask_vertical = crop_filename_rgb_mask.replace("_mask.png", "_vertical_mask.png")

                image_crop_vertical.save(crop_filename_image_vertical)
                grayscale_mask_crop_vertical.save(crop_filename_grayscale_mask_vertical)
                rgb_mask_crop_vertical.save(crop_filename_rgb_mask_vertical)

                # Apply horizontal flip and save
                image_crop_horizontal = image_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                grayscale_mask_crop_horizontal = grayscale_mask_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                rgb_mask_crop_horizontal = rgb_mask_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

                crop_filename_image_horizontal = crop_filename_image.replace(".jpg", "_horizontal.jpg")
                crop_filename_grayscale_mask_horizontal = crop_filename_grayscale_mask.replace("_mask.png", "_horizontal_mask.png")
                crop_filename_rgb_mask_horizontal = crop_filename_rgb_mask.replace("_mask.png", "_horizontal_mask.png")

                image_crop_horizontal.save(crop_filename_image_horizontal)
                grayscale_mask_crop_horizontal.save(crop_filename_grayscale_mask_horizontal)
                rgb_mask_crop_horizontal.save(crop_filename_rgb_mask_horizontal)

        return image_crops, grayscale_mask_crops, rgb_mask_crops


if __name__ == "__main__":
    """ 
     Args:
        base_dir (string):  dataset base directory path to be updated """
    
    base_dir = r"D:\Semesterarbeit\lane-marking-detection\DeepLabV3Plus\skyscapes\\"

    # Training set directories
    train_image_dir = base_dir + r"train\images\\"
    train_mask_dir_rgb = base_dir + r"train\labels\rgb\\"
    train_mask_dir_grayscale = base_dir + r"train\labels\grayscale\\"
    train_crop_save_image_dir = base_dir + r"augmented_data\train\images\\"
    train_crop_save_grayscale_mask_dir = base_dir + r"augmented_data\train\labels\grayscale\\"
    train_crop_save_rgb_mask_dir = base_dir + r"augmented_data\train\labels\rgb\\"

    # Validation set directories
    val_image_dir = base_dir + r"val\images\\"
    val_mask_dir_rgb = base_dir + r"val\labels\rgb\\"
    val_mask_dir_grayscale = base_dir + r"val\labels\grayscale\\"
    val_crop_save_image_dir = base_dir + r"augmented_data\val\images\\"
    val_crop_save_grayscale_mask_dir = base_dir + r"augmented_data\val\labels\grayscale\\"
    val_crop_save_rgb_mask_dir = base_dir + r"augmented_data\val\labels\rgb\\"


    overlap_percentage_50 = 0.5 # Overlap percentage could be changed here
    overlap_percentage_10 = 0.1 # Overlap percentage could be changed here

    # Check and create directories if needed for training set
    for directory in [train_crop_save_image_dir, train_crop_save_grayscale_mask_dir, train_crop_save_rgb_mask_dir]:
        os.makedirs(directory, exist_ok=True)

    # Check and create directories if needed for validation set
    for directory in [val_crop_save_image_dir, val_crop_save_grayscale_mask_dir, val_crop_save_rgb_mask_dir]:
        os.makedirs(directory, exist_ok=True)
    

    # For training set
    transform_50_train_rgb = SlidingWindowCrop((512, 512), train_crop_save_image_dir, train_crop_save_grayscale_mask_dir, train_crop_save_rgb_mask_dir, overlap=overlap_percentage_50)
    transform_10_train_rgb = SlidingWindowCrop((512, 512), train_crop_save_image_dir, train_crop_save_grayscale_mask_dir, train_crop_save_rgb_mask_dir, overlap=overlap_percentage_10)

    transform_50_train_grayscale = SlidingWindowCrop((512, 512), train_crop_save_image_dir, train_crop_save_grayscale_mask_dir, train_crop_save_rgb_mask_dir, overlap=overlap_percentage_50)
    transform_10_train_grayscale = SlidingWindowCrop((512, 512), train_crop_save_image_dir, train_crop_save_grayscale_mask_dir, train_crop_save_rgb_mask_dir, overlap=overlap_percentage_10)


    train_dataset_rgb = SkyscapesDataset(
        train_image_dir, train_mask_dir_grayscale, train_mask_dir_rgb,
        transform=transform_50_train_rgb, transform_grayscale_mask=transform_50_train_grayscale, transform_rgb_mask=transform_50_train_rgb
    )

    # Iterate through the training dataset and save the cropped images with 50% overlap
    for i, (image_crops, grayscale_mask_crops, rgb_mask_crops) in enumerate(train_dataset_rgb):
        pass  # Images are already saved in the specified directory

    # Iterate through the training dataset again and save the cropped images with 10% overlap
    train_dataset_rgb.transform = transform_10_train_rgb
    for i, (image_crops, grayscale_mask_crops, rgb_mask_crops) in enumerate(train_dataset_rgb):
        pass  # Images are already saved in the specified directory

    # For validation set
    transform_50_val_rgb = SlidingWindowCrop((512, 512), val_crop_save_image_dir, val_crop_save_grayscale_mask_dir, val_crop_save_rgb_mask_dir, overlap=overlap_percentage_50)
    transform_10_val_rgb = SlidingWindowCrop((512, 512), val_crop_save_image_dir, val_crop_save_grayscale_mask_dir, val_crop_save_rgb_mask_dir, overlap=overlap_percentage_10)

    transform_50_val_grayscale = SlidingWindowCrop((512, 512), val_crop_save_image_dir, val_crop_save_grayscale_mask_dir, val_crop_save_rgb_mask_dir, overlap=overlap_percentage_50)
    transform_10_val_grayscale = SlidingWindowCrop((512, 512), val_crop_save_image_dir, val_crop_save_grayscale_mask_dir, val_crop_save_rgb_mask_dir, overlap=overlap_percentage_10)

    val_dataset_rgb = SkyscapesDataset(
        val_image_dir, val_mask_dir_grayscale, val_mask_dir_rgb,
        transform=transform_50_val_rgb, transform_grayscale_mask=transform_50_val_grayscale, transform_rgb_mask=transform_50_val_rgb
    )

    # Iterate through the validation dataset and save the cropped images with 50% overlap
    for i, (image_crops, grayscale_mask_crops, rgb_mask_crops) in enumerate(val_dataset_rgb):
        pass  # Images are already saved in the specified directory

    # Iterate through the validation dataset again and save the cropped images with 10% overlap
    val_dataset_rgb.transform = transform_10_val_rgb
    for i, (image_crops, grayscale_mask_crops, rgb_mask_crops) in enumerate(val_dataset_rgb):
        pass  # Images are already saved in the specified directory
