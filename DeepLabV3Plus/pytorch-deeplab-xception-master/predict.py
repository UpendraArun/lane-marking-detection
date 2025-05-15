import os
from datetime import datetime
import torch
from PIL import Image
import numpy as np
import torch.nn.init as init
from torchvision import transforms
from modeling.deeplab import DeepLab

@torch.no_grad()
def predict(model_checkpoint, input_folder, output_folder, detection_class, backbone):
    
    """ Args:
        model_checkpoint (string): path to model checkpoint
        input_folder (string): path to the folder containing input images
        output_folder (string): path to save the prediction masks.
        detection_class (int): 1 for multi-class, 2 for binary class  """

   
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]

    # Load the model
    if detection_class == 1:
        num_classes = 13
    elif detection_class == 2:
        num_classes = 2
    
    model = DeepLab(backbone=backbone, num_classes=num_classes) # Change backbone here
    model.eval()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=lambda storage, loc: storage))

    # Create output folder if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_folder = os.path.join(output_folder, f"trial_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            out_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_output.png")

            image = Image.open(image_path)
            image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD)])
            image_tensor = image_transforms(image)[None].to(device=DEVICE, dtype=torch.float)

            out = model(image_tensor)[0].squeeze()
            out_max = out.max(0, keepdim=False)[1].cpu().numpy()

            final_image = np.zeros((out_max.shape[0], out_max.shape[1], 3), dtype=np.uint8)
            if detection_class == 1:
                final_image[(out_max == 0), :] = np.array([ 255, 255,  255])  #class 0 
                final_image[(out_max == 1), :] = np.array([255,    0,    0])  #class 1
                final_image[(out_max == 2), :] = np.array([  0,    0,  255])  #class 2
                final_image[(out_max == 3), :] = np.array([255,  255,    0])  #class 3
                final_image[(out_max == 4), :] = np.array([  0,  255,    0])  #class 4
                final_image[(out_max == 5), :] = np.array([255,  128,    0])  #class 5
                final_image[(out_max == 6), :] = np.array([128,    0,    0])  #class 6
                final_image[(out_max == 7), :] = np.array([  0,  255,  255])  #class 7
                final_image[(out_max == 8), :] = np.array([  0,  128,    0])  #class 8
                final_image[(out_max == 9), :] = np.array([255,    0,  255])  #class 9
                final_image[(out_max == 10), :] = np.array([  0,  150,  150]) #class 10
                final_image[(out_max == 11), :] = np.array([200,  200,    0]) #class 11
                final_image[(out_max == 12), :] = np.array([100,    0,  200]) #class 12
            elif detection_class == 2:
                final_image[(out_max == 0), :] = np.array([ 0,      0,    0])  #class 0 
                final_image[(out_max == 1), :] = np.array([255,   255,  255])  #class 1


            final_image_pil = Image.fromarray(final_image)
            final_image_pil.save(out_file)

if __name__ == "__main__":
    
    model_checkpoint = r"D:\Semesterarbeit\lane-marking-detection\DeepLabV3Plus\Trials\Trial_21\model_best_18percent.pth" # Change this to the path of your saved model configuration
    input_folder = r"D:\Semesterarbeit\lane-marking-detection\DeepLabV3Plus\Prediction_Images"  # Change this to the path of your input image folder
    output_folder = r"./output"  # Change this to the desired output folder
    detection_class = 1
    backbone = 'resnet'

    print("Absolute path to the input folder:", os.path.abspath(input_folder))
    print("Absolute path to the output folder:", os.path.abspath(output_folder))

    predict(model_checkpoint, input_folder, output_folder, detection_class, backbone)
