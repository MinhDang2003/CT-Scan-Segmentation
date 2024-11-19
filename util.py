from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from datetime import datetime
from tqdm import tqdm

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def train(
    model,
    train_loader,
    val_loader,
    loss,
    optim,
    schedule,
    max_epochs,
    model_dir,
    device,
    val_interval = 1
):
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.makedirs(model_dir, exist_ok=True)
    path_saved = os.path.join(model_dir, current_time)
    os.makedirs(path_saved)
    
    print(f"Begin training at: {current_time}")
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_val = []
    save_metric_train = []
    save_metric_val = []

    for epoch in range(max_epochs):
        print("-" * 40)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        
        for batch_data in train_loader:
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            # label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(volume)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'* 40)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(path_saved, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(path_saved, 'metric_train.npy'), save_metric_train)

        schedule.step()
        
        print(f"Learning rate for epoch {epoch+1}: {schedule.get_last_lr()[0]}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                val_metric = 0
                epoch_metric_val = 0
                val_step = 0

                for val_data in val_loader:
                    val_step += 1

                    val_volume = val_data["vol"]
                    val_label = val_data["seg"]
                    # val_label = val_label != 0
                    val_volume, val_label = (val_volume.to(device), val_label.to(device),)
                    
                    val_outputs = model(val_volume)
                    
                    val_loss = loss(val_outputs, val_label)
                    val_epoch_loss += val_loss.item()
                    val_metric = dice_metric(val_outputs, val_label)
                    epoch_metric_val += val_metric
                    
                
                val_epoch_loss /= val_step
                print(f'val_loss_epoch: {val_epoch_loss:.4f}')
                save_loss_val.append(val_epoch_loss)
                np.save(os.path.join(path_saved, 'loss_val.npy'), save_loss_val)

                epoch_metric_val /= val_step
                print(f'val_dice_epoch: {epoch_metric_val:.4f}')
                save_metric_val.append(epoch_metric_val)
                np.save(os.path.join(path_saved, 'metric_val.npy'), save_metric_val)

                if epoch_metric_val > best_metric:
                    best_metric = epoch_metric_val
                    best_metric_epoch = epoch + 1
 
                    model_filename = f"{best_metric:.4f}.pth"
                    torch.save(model.state_dict(), os.path.join(
                        path_saved, model_filename))
                
                print(
                    f"{'-' * 40}"
                    f"\ncurrent epoch: {epoch + 1} current mean dice: {val_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


    print(
        f"train completed at {datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
