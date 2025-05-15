import os
import cv2
import numpy as np

#################################################################################
#       Parameters to be updated here
#################################################################################
""" Args:
        Interpolation method (int): Interpolation method
                                    1 - nearest neighbour interpolation
                                    2 - bilinear interpolation
                                    3 - cubic interpolation
                                    4 - Lanczos interpolation
        target gsd (int): intended Ground Sampling Distance in cm
        source_folder (string): source directory containing bavarian orthophotos 
        output_folder (string): output directory where the resized images will be stored """
 
interpolation_method = 1  
target_gsd = 13
source_folder = r'D:\Semesterarbeit\lane-marking-detection\DeepLabV3Plus\Bavarian_orthophotos'
output_folder = r'D:\Semesterarbeit\lane-marking-detection\DeepLabV3Plus\output_resized_images'
#################################################################################

os.makedirs(output_folder, exist_ok=True)

if interpolation_method == 1:
    interpolation = cv2.INTER_NEAREST
elif interpolation_method == 2:
    interpolation = cv2.INTER_LINEAR
elif interpolation_method == 3:
    interpolation = cv2.INTER_CUBIC
elif interpolation_method == 4:
    interpolation = cv2.INTER_LANCZOS4


def resize_image(image, target_gsd, current_gsd, interpolation):
    # Calculate the scale factor needed to achieve the target GSD
    scale_factor = current_gsd / target_gsd

    # Resize the image using bilinear interpolation
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation)

    # Ensure the values are within the valid range for images
    resized_image = np.clip(resized_image, 0, 255).astype(np.uint8)

    return resized_image

def process_images(source_folder, output_folder, target_gsd):
    # Iterate through all image files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Load the original image
            image_path = os.path.join(source_folder, file_name)
            original_image = cv2.imread(image_path)

            # Calculate the current GSD (assuming square pixels)
            height, width, _ = original_image.shape
            current_gsd = max(height, width) / 40  # Assuming square pixels

            # Resize the image directly
            resized_image = resize_image(original_image, target_gsd, current_gsd)

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, f'Resized_NN_{file_name}')
            cv2.imwrite(output_path, resized_image)

            print(f"Resized {file_name} to achieve {target_gsd} cm GSD.") 

# Process all images in the source folder and save resized images to the output folder
process_images(source_folder, output_folder, target_gsd, interpolation)