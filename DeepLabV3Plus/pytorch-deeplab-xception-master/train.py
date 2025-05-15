#train.py

import time
import os
import numpy as np
import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from modeling.deeplab import *
from dataloaders.datasets import skyscapes as skyscapes2
from dataloaders.datasets import skyscapesbw
from sklearn.metrics import confusion_matrix
import torch.nn.init as init
from loss_func.ce_iou_loss import CombinedLoss
from loss_func.ce_dice_loss import CombinedLossCEAndDice
from torch.utils.tensorboard import SummaryWriter


#function to calculate the number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#function to calculate the confusion matrix
def compute_iou(confusion_matrix):
    epsilon = 1e-10
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - intersection
    iou = intersection / (union + epsilon)
    return iou

""" function to calculate the weights required 
to calculate frequency weighted IoU """
def calculate_weigths_labels(confusion_matrix):
    epsilon = 1e-10
    class_frequencies = np.sum(confusion_matrix, axis=1) 
    class_weights = class_frequencies / (np.sum(class_frequencies) + epsilon)
    return class_weights

# Start logging time
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


#################################################################################
#       Parameters to be updated here
#################################################################################
""" Args:
        trial_number (int): experiment number if you wish to use
        backbone (string): backbone of the model
        detection_class (int): 1 for multi-class detection, 2 for binary detection 
        num_workers (int): total number of workers
        He_Uniform (string): 1 if HeUniform Initialization is to be applied, 0 for not application
        loss_func (int): 1 for cross entropy loss, 2 for cross entropy with IoU Loss, 3 for cross entropy with Dice loss
        
        num_epochs (int): number of epochs
        log_nth (int): logging frequency for loss and accuracy.
        batch_size (int): Batch size for training and validation
        class weights_multi_class (float): row matrix of class weights for loss function multi-class
        class_weights_binary (float): row matrix of class weights for loss function binary case
        patience (int): patience for early stopping of training    """

trial_number = 1  # adjust this based on your trials/ required only for traceability
backbone = 'resnet'
detection_class = 1
num_workers = 4
He_Uniform = 0
loss_func = 2


num_epochs = 60
log_nth = 50  # log_nth: log training accuracy and loss every nth iteration
batch_size = 32
class_weights_multi_class = [1.0000, 9.0000, 8.0000, 9.0000, 10.0000, 10.0000, 11.0000, 10.0000, 10.0000, 9.0000, 10.0000, 10.0000, 10.0000]
class_weights_binary = [1.0000, 9.0000] 
patience = 10  
#################################################################################

# Load the model
if detection_class == 1:
    num_classes = 13
elif detection_class == 2:
    num_classes = 2

else:
    print("Correct detection_class not selected. Automatically starting with multi-class detection")

model = DeepLab(backbone=backbone, output_stride=16, num_classes=num_classes, sync_bn=True, freeze_bn=False)

# Calculate and print the number of parameters of the model
total_params = count_parameters(model)
print(f'Total number of parameters in the model: {total_params}')

# Initialize the model weights with HeUniform
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # Apply HeUniform initialization to Conv2d layers
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.ConvTranspose2d):
        # Apply HeUniform initialization to ConvTranspose2d layers
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm2d layers with a constant weight of 1 and bias of 0
        if hasattr(m, 'weight') and m.weight is not None:
            init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias, 0)
            

# Apply the HeUniform initialization to the model
if He_Uniform > 0:
    model.apply(weights_init)


# Initialize the Dataset
if num_classes == 13:
    train_set = skyscapes2.SkyscapesDataset(split='train')
    val_set = skyscapes2.SkyscapesDataset(split='val')
    #test_set = skyscapes2.SkyscapesDataset(split='test')
    class_weights = torch.tensor(class_weights_multi_class)
elif num_classes == 2:
    train_set = skyscapesbw.SkyscapesDataset(split='train')
    val_set = skyscapesbw.SkyscapesDataset(split='val')
    #test_set = skyscapesbw.SkyscapesDataset(split='test')
    class_weights = torch.tensor(class_weights_binary)


# Load the Dataset 
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)




print("Shape of class_weights:", class_weights.shape)
print("Values of class_weights:", class_weights)

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Value to determine the amount of soft iou/dice loss component to the combined loss
iou_dice_weight = 1.0

# Initializing Loss function
if loss_func == 1:
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
elif loss_func == 2:
    loss_func = CombinedLoss(weight_ce=class_weights.to(device), weight_iou=iou_dice_weight).to(device)
elif loss_func == 3:
    loss_func = CombinedLossCEAndDice(weight_ce=class_weights.to(device), weight_dice=iou_dice_weight).to(device)


# Initialize performance metrics
train_loss_history = []
train_acc_history = []
val_acc_history = []
val_loss_history = []


model.to(device)
# Define the optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)

iter_per_epoch = len(train_loader)
print('iterations per epoch: ',iter_per_epoch)


# Check for existing checkpoints
check_pth = 'checkpoints'
os.makedirs(check_pth, exist_ok=True)
checkpoint_path = 'checkpoints/checkpoint.pth'
model_best = 'model_best'
os.makedirs(model_best, exist_ok=True)

if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_miou = checkpoint['best_miou']
    current_patience = checkpoint['current_patience']
    print(f"Resuming training from epoch {epoch + 1}")

else:
    epoch = 0
    best_miou = 0
    current_patience = 0
    print('Start training.')

# TRAINING LOOP STARTS
for epoch in range(epoch, num_epochs):
    
    train_acc_epoch = []
    
    for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        train_loss_history.append(loss.item())

        if log_nth and i % log_nth == 0:
            last_log_nth_losses = train_loss_history[-log_nth:]
            train_loss = np.mean(last_log_nth_losses)
            print('[Iteration %d/%d] TRAIN loss: %.3f' % (i + epoch * iter_per_epoch,iter_per_epoch * num_epochs,train_loss))
            
            # Log training loss to TensorBoard
            writer.add_scalar('Loss/Train', train_loss, i + epoch * iter_per_epoch)

        _, preds = torch.max(outputs, 1)

        targets_mask = targets >= 0            
        
        # Calculate training accuracy
        train_acc = torch.mean((preds == targets)[targets_mask].float())
        train_acc_history.append(train_acc.item())
        train_acc_epoch.append(train_acc.item())
        

    if log_nth:
        train_acc =  np.mean(train_acc_epoch)
        print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, train_acc, train_loss))

    
    # VALIDATION LOOP STARTS
    val_losses = []
    val_scores = []
    model.eval()
    confusion_matrix_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model.forward(inputs)
            loss = loss_func(outputs, targets)
            val_losses.append(loss.detach().cpu().numpy())

            _, preds = torch.max(outputs, 1)

            targets_mask = targets >= 0

            targets_np = targets[targets_mask].cpu().numpy()
            preds_np = preds[targets_mask].cpu().numpy()

            confusion_matrix_total += confusion_matrix(targets_np, preds_np, labels=np.arange(num_classes))
                
            scores = np.mean(targets_np == preds_np)
            val_scores.append(scores)


    model.train()

    # Calculate mean IoU
    iou = compute_iou(confusion_matrix_total)
    miou = np.nanmean(iou)

    # Calculate validation accuracy/pixel accuracy
    val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
    val_acc_history.append(val_acc)
    val_loss_history.append(val_loss)
    if log_nth:
        print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, val_acc, val_loss))
        print(f'Mean IoU (mIoU) for all classes: {miou:.3f}')
        
        # Log validation loss and mean IoU
        writer.add_scalar('Loss/Validation', val_loss, epoch+1)
        writer.add_scalar('mIoU/Validation', miou, epoch+1)
        
        # Log class wise mean IoU
        print('IoU for each class:')
        for class_idx in range(num_classes):
            print(f'Class {class_idx}: {iou[class_idx]:.3f}')
            writer.add_scalar(f'Class_IoU/{class_idx}', iou[class_idx], epoch+1)
        
        # Calculate frequency-weighted IoU
        weight_labels = calculate_weigths_labels(confusion_matrix_total)
        freq_weighted_iou = np.sum(weight_labels[weight_labels>0] * iou[weight_labels>0])
        print(f'Frequency-Weighted IoU: {freq_weighted_iou:.3f}')
        
        # Log frequency-weighted IoU
        writer.add_scalar('Frequency_Weighted_IoU/Validation', freq_weighted_iou, epoch+1)

        # Calculate precision, recall, and average precision using the confusion matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.nan_to_num(np.diag(confusion_matrix_total) / np.sum(confusion_matrix_total, axis=0), nan=0)
            recall = np.nan_to_num(np.diag(confusion_matrix_total) / np.sum(confusion_matrix_total, axis=1), nan=0)

        # Calculate Average precision calculation
        avg_precision = np.nanmean(precision)

        # Calculate Average recall calculation
        avg_recall = np.nanmean(recall)

        # Print and log precision and recall
        print(f'Average Precision: {avg_precision:.3f}')
        print(f'Average Recall: {avg_recall:.3f}')

        writer.add_scalar('Average_Precision/Validation', avg_precision, epoch+1)
        writer.add_scalar('Average_Recall/Validation', avg_recall, epoch+1)

        # Empty cache
        torch.cuda.empty_cache()

    
    # Check for early stopping
    if miou > best_miou:
        best_miou = miou
        current_patience = 0
        # Save the current best model
        torch.save(model.state_dict(), f"model_best/model_best_trial_{trial_number}_epoch_{epoch + 1}.pth")

        # Save checkpoint for training interruptions
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'current_patience': current_patience
        }
        torch.save(checkpoint, checkpoint_path)
    else:
        current_patience += 1
        print('current patience is: ',current_patience)
        if current_patience >= patience:
            print(f'Early stopping at epoch {epoch + 1} as there is no improvement in mIoU for the last {patience} epochs.')
            break

end_time = time.time()
print(f'Total training time with {epoch + 1} epochs is {(end_time - start_time)/60:.2f} minutes')
writer.close()