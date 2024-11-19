import os
import numpy as np
from glob import glob
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
from config import load_config

transform_map = {
    "LoadImaged": LoadImaged,
    "EnsureChannelFirstD": EnsureChannelFirstD,
    "Spacingd": Spacingd,
    "Orientationd": Orientationd,
    "ScaleIntensityRanged": ScaleIntensityRanged,
    "CropForegroundd": CropForegroundd,
    "Resized": Resized,
    'ToTensord': ToTensord
}

def create_transforms(config_transforms):
    transforms = []
    for transform_cfg in config_transforms:
        # Get the transform name and params
        transform_name = transform_cfg["name"]
        params = transform_cfg.get("params", {})
        if params is None:
            params = {}
        if transform_name not in ['ScaleIntensityRanged']:
            params['keys'] = ['vol','seg']
        else:
            params['keys'] = ['vol']
        if transform_name in ['CropForegroundd']:
            params['source_key'] = 'vol'
        # if transform_name in ['Resized']:
        #     params['spatial_size'] = spatial_size
        # Create the transform with parameters
        transform_class = transform_map.get(transform_name)
        if transform_class is not None:
            transforms.append(transform_class(**params))
        else:
            print(f"Warning: Transform {transform_name} not recognized.")
    return (transforms)

def create_data_loaders(
    image_label_pairs, 
    val_ratio=0.15, 
    test_ratio=0.05, 
    seed=42, 
    determinism_seed=0,
    train_batch_size=8, 
    val_test_batch_size=32,
    cache=True,
    train_transform=None,
    val_transform=None,
    test_transform=None):
    
    
    train_pairs, test_pairs = train_test_split(image_label_pairs, test_size=test_ratio, random_state=seed)
    
    val_ratio_adjusted = val_ratio / ((1 - (val_ratio + test_ratio)) + val_ratio)
    
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=val_ratio_adjusted, random_state=seed)
    
    set_determinism(seed=determinism_seed)
    
    DatasetClass = CacheDataset if cache else Dataset
    train_ds = DatasetClass(data=train_pairs, transform=train_transform, cache_rate=1.0)
    val_ds = DatasetClass(data=val_pairs, transform=val_transform, cache_rate=1.0)
    test_ds = DatasetClass(data=test_pairs, transform=test_transform, cache_rate=1.0)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size)
    val_loader = DataLoader(val_ds, batch_size=val_test_batch_size)
    test_loader = DataLoader(test_ds, batch_size=val_test_batch_size)
    
    return train_loader, val_loader, test_loader

def load_data(arg):
    # Main execution
    base_path = arg['paths']['data_dir']

    volumes = glob(os.path.join(base_path, "images", '*.nii.gz'))
    segmentation = glob(os.path.join(base_path, "labels", '*.nii.gz'))
    # print(volumes)
    # print(segmentation)
    # return
    
    files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(volumes, segmentation)]
    
    # Create data loaders
    val_ratio = arg['dataset']['split_ratio']['val']
    test_ratio = arg['dataset']['split_ratio']['test']
    seed = arg['dataset']['seed']
    train_batch_size = arg['dataloader']['train_batch_size']
    val_test_batch_size = arg['dataloader']['val_test_batch_size']
    
    train_transforms = Compose(create_transforms(arg["transforms"]["train"]))
    # val_transforms = Compose(create_transforms(arg["transforms"]["train"]))
    test_transforms = Compose(create_transforms(arg["transforms"]["test"]))
    
    train_loader, val_loader, test_loader = create_data_loaders(
        files,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        determinism_seed=arg['dataloader']['determinism_seed'],
        train_batch_size=train_batch_size,
        val_test_batch_size=val_test_batch_size,
        cache=arg['dataloader']['cache'],
        train_transform=train_transforms,
        val_transform=train_transforms,
        test_transform=test_transforms
        )

    # Print the sizes of each split
    print("Number of training pairs:", len(train_loader.dataset))
    print("Number of validation pairs:", len(val_loader.dataset))
    print("Number of testing pairs:", len(test_loader.dataset))
    
    return train_loader, val_loader, test_loader

# train_loader, val_loader, test_loader = prepare_data(load_config())

import matplotlib.pyplot as plt
import numpy as np

def visualize_data(loader, num_samples=5):
    """
    Visualizes images and their corresponding labels side by side.
    
    Args:
        loader: The data loader to sample from.
        num_samples: The number of samples to visualize.
    """
    for idx, batch in enumerate(loader):
        if idx >= num_samples:
            break
        # Assuming the loader returns a dictionary with keys 'vol' and 'seg'
        images = batch['vol'].numpy()  # Convert tensors to numpy arrays
        labels = batch['seg'].numpy()  # Convert tensors to numpy arrays

        # Display each sample in the batch
        batch_size = images.shape[0]
        for i in range(min(batch_size, num_samples)):
            img = images[i, 0, : , : ,0]  # Assuming the first channel is the grayscale volume
            lbl = labels[i, 0, : , : ,0]  # Assuming the first channel is the label
            
            plt.figure(figsize=(10, 5))
            
            # Show the image
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='bone')
            plt.title("Image")
            plt.axis('off')
            
            # Show the label
            plt.subplot(1, 2, 2)
            plt.imshow(lbl, cmap='bone')
            plt.title("Label")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
# visualize_data(train_loader)
