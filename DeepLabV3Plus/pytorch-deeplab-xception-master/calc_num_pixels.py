#calc_num_pixels.py This is used only for multi-class

import time
import os
import numpy as np
import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from modeling.deeplab import *
from dataloaders.datasets import skyscapes as skyscapes2
from sklearn.metrics import confusion_matrix
import torch.autograd.profiler as profiler

from loss_func.ce_iou_loss import CombinedLoss
from loss_func.ce_dice_loss import CombinedLossCEAndDice
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter


#################################################################################
#       Parameters to be updated here
#################################################################################
""" Args:        
        num_epochs (int): number of epochs
        log_nth (int): logging frequency for loss and accuracy.
        batch_size (int): Batch size    """

num_epochs = 1  # 1 Epoch is sufficient to get to know the number of pixels per class in the dataset
log_nth = 50  # log_nth: log training accuracy and loss every nth iteration
batch_size = 32 # Vary batch size based on the GPU capacity
 
#################################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_iou(confusion_matrix):
    epsilon = 1e-10
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - intersection
    iou = intersection / (union + epsilon)
    return iou

def calculate_weigths_labels(confusion_matrix):
    # Calculate class weights as the inverse of class frequencies
    epsilon = 1e-10
    class_frequencies = np.sum(confusion_matrix, axis=1) 
    class_weights = class_frequencies / (np.sum(class_frequencies) + epsilon)

    return class_weights

start_time = time.time()

# Create a main 'logs' directory if it doesn't exist
main_log_dir = 'logs'
os.makedirs(main_log_dir, exist_ok=True)

# Create a subdirectory with a timestamp for each run
log_subdir = time.strftime("%Y%m%d%H%M%S")
log_dir = os.path.join(main_log_dir, log_subdir)
os.makedirs(log_dir, exist_ok=True)

# Create a SummaryWriter for TensorBoard logging
writer = SummaryWriter(log_dir)

# Load the model
model = DeepLab(backbone='xception', output_stride=16, num_classes=13, sync_bn=True, freeze_bn=False)


total_params = count_parameters(model)
print(f'Total number of parameters in the model: {total_params}')


# Initialize the Dataset
train_set = skyscapes2.SkyscapesDataset(split='train')
val_set = skyscapes2.SkyscapesDataset(split='val')
test_set = skyscapes2.SkyscapesDataset(split='test')
#num_class = train_set.NUM_CLASSES
num_class = 13



# Load the Dataset 
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Define Loss function 
num_classes = 13  # Update this based on your actual number of classes
background_weight = 0.5  # Adjust this as needed

class_weights = torch.ones(num_classes)
class_weights[0] = background_weight  # Background class weight

# Normalize the weights to ensure they sum to 1
class_weights /= class_weights.sum()
class_weights *= 10


# Now you can use class_weights in nn.CrossEntropyLoss


print("Shape of class_weights:", class_weights.shape)
print("Values of class_weights:", class_weights)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iou_dice_weight = 1.0
#loss_func = torch.nn.CrossEntropyLoss().to(device)
#loss_func = CombinedLoss(weight_ce=class_weights.to(device), weight_iou=iou_dice_weight).to(device)
loss_func = CombinedLossCEAndDice(weight_ce=class_weights.to(device), weight_dice=iou_dice_weight).to(device)


#Initialize performance metrics
train_loss_history = []
train_acc_history = []
val_acc_history = []
val_loss_history = []


model.to(device)
# Define the optimizer and the hyper parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)

iter_per_epoch = len(train_loader)
print('iterations per epoch: ',iter_per_epoch)


#device = torch.device("cpu")

# Check for existing checkpoints
check_pth = 'checkpoints'
os.makedirs(check_pth, exist_ok=True)
checkpoint_path = 'checkpoints/checkpoint.pth'

epoch = 0
best_miou = 0
current_patience = 0
print('Start training.')


patience = 20
#accumulation_steps = 2

for epoch in range(num_epochs):
    # TRAINING
    train_acc_epoch = []
    #torch.cuda.empty_cache()
    for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        #logits = outputs['out']
        #targets = targets.long()  # Ensure targets are of type LongTensor
        #loss = loss_func(logits, targets)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        #if i % accumulation_steps == 0:
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        #optimizer.step()
        optimizer.step()

        #train_loss_history.append(loss.cpu().detach().numpy())
        train_loss_history.append(loss.item())

        if log_nth and i % log_nth == 0:
            last_log_nth_losses = train_loss_history[-log_nth:]
            train_loss = np.mean(last_log_nth_losses)
            print('[Iteration %d/%d] TRAIN loss: %.3f' % (i + epoch * iter_per_epoch,iter_per_epoch * num_epochs,train_loss))

            writer.add_scalar('Loss/Train', train_loss, i + epoch * iter_per_epoch)

        _, preds = torch.max(outputs, 1)

        targets_mask = targets >= 0            
        train_acc = torch.mean((preds == targets)[targets_mask].float())
        train_acc_history.append(train_acc.item())
        train_acc_epoch.append(train_acc.item())

        

    if log_nth:
        train_acc =  np.mean(train_acc_epoch)
        print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, train_acc, train_loss))

    # Assuming confusion_matrix_total is the confusion matrix obtained after training
    confusion_matrix_total = np.zeros((num_class, num_class), dtype=np.int64)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model.forward(inputs)

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with target >= 0 e.g. for segmentation
            targets_mask = targets >= 0

            # Convert targets and preds to numpy arrays
            targets_np = targets[targets_mask].cpu().numpy()
            preds_np = preds[targets_mask].cpu().numpy()

            confusion_matrix_total += confusion_matrix(targets_np, preds_np, labels=np.arange(num_class))

    # Set the model back to training mode
    model.train()

   # Extract the total pixels for each class
    total_pixels_train = np.sum(confusion_matrix_total, axis=1)

    # Save the details in a text file
    class_stats_path = 'class_stats'
    os.makedirs(class_stats_path, exist_ok=True)
    output_file_path_pixels_train = 'class_stats/training_class_pixels.txt'

    with open(output_file_path_pixels_train, 'w') as f:
        f.write("Class\tTotal Pixels\n")
        for class_idx, total_pixels_class in enumerate(total_pixels_train):
            f.write(f"{class_idx}\t{total_pixels_class}\n")

    print(f"Training class pixels details saved to {output_file_path_pixels_train}")


        
    # VALIDATION
    val_losses = []
    val_scores = []
    model.eval()
    confusion_matrix_total = np.zeros((num_class, num_class), dtype=np.int64)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model.forward(inputs)
            loss = loss_func(outputs, targets)
            val_losses.append(loss.detach().cpu().numpy())

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with target >= 0 e.g. for
            # segmentation
            targets_mask = targets >= 0

            # Convert targets and preds to numpy arrays
            targets_np = targets[targets_mask].cpu().numpy()
            preds_np = preds[targets_mask].cpu().numpy()

            confusion_matrix_total += confusion_matrix(targets_np, preds_np, labels=np.arange(num_class))
                
            # Calculate accuracy using numpy arrays
            scores = np.mean(targets_np == preds_np)
            val_scores.append(scores)


    model.train()

    # Calculate and print mean IoU
    iou = compute_iou(confusion_matrix_total)
    miou = np.nanmean(iou)

    # Extract the total pixels for each class
    total_pixels_val = np.sum(confusion_matrix_total, axis=1)

    # Save the details in a text file
    output_file_path_pixels_val = 'class_stats/validation_class_pixels.txt'

    with open(output_file_path_pixels_val, 'w') as f:
        f.write("Class\tTotal Pixels\n")
        for class_idx, total_pixels_class in enumerate(total_pixels_val):
            f.write(f"{class_idx}\t{total_pixels_class}\n")

    print(f"Validation class pixels details saved to {output_file_path_pixels_val}")





    val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
    val_acc_history.append(val_acc)
    val_loss_history.append(val_loss)
    if log_nth:
        print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, val_acc, val_loss))
        print(f'Mean IoU (mIoU) for all classes: {miou:.3f}')
        
        # Log validation loss and mIoU
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('mIoU/Validation', miou, epoch)
        
        # Log class wise mIoU
        print('IoU for each class:')
        for class_idx in range(num_class):
            print(f'Class {class_idx}: {iou[class_idx]:.3f}')
            writer.add_scalar(f'Class_IoU/{class_idx}', iou[class_idx], epoch)
        
        # Calculate frequency-weighted IoU
        weight_labels = calculate_weigths_labels(confusion_matrix_total)
        freq_weighted_iou = np.sum(weight_labels[weight_labels>0] * iou[weight_labels>0])
        print(f'Frequency-Weighted IoU: {freq_weighted_iou:.3f}')
        
        # Log f.w.IoU
        writer.add_scalar('Frequency_Weighted_IoU/Validation', freq_weighted_iou, epoch)

        # Calculate precision, recall, and average precision using the confusion matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.nan_to_num(np.diag(confusion_matrix_total) / np.sum(confusion_matrix_total, axis=0), nan=0)
            recall = np.nan_to_num(np.diag(confusion_matrix_total) / np.sum(confusion_matrix_total, axis=1), nan=0)

        # Average precision calculation
        avg_precision = np.nanmean(precision)

        # Average recall calculation
        avg_recall = np.nanmean(recall)

        # Print and log precision and recall
        print(f'Average Precision: {avg_precision:.3f}')
        print(f'Average Recall: {avg_recall:.3f}')

        writer.add_scalar('Average_Precision/Validation', avg_precision, epoch)
        writer.add_scalar('Average_Recall/Validation', avg_recall, epoch)


        
        torch.cuda.empty_cache()

    
    # Check for early stopping
    if miou > best_miou:
        best_miou = miou
        current_patience = 0
        # Save the current best model
        torch.save(model.state_dict(), "model_best.pth")

     
    else:
        current_patience += 1
        print('current patience is: ',current_patience)
        if current_patience >= patience:
            print(f'Early stopping at epoch {epoch + 1} as there is no improvement in mIoU for the last {patience} epochs.')
            break

end_time = time.time()
print(f'Total training time with {epoch + 1} epochs is {(end_time - start_time)/60:.2f} minutes')
writer.close()