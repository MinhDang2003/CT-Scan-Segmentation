from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
import monai

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from load_data import load_data
from config import load_config
from util import train

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU with CUDA
    device_type = "CUDA (GPU)"
elif torch.backends.mps.is_built():
    device = torch.device("mps")  # Use Apple MPS
    device_type = "MPS (Metal Performance Shaders)"
else:
    device = torch.device("cpu")  # Default to CPU
    device_type = "CPU"

# Display the selected device
print(f"Using device: {device_type} ({device})")

config = load_config()

train_loader, val_loader, test_loader = load_data(config)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,    
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, amsgrad=True)

scheduler = CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-5)  # T_max is the number of epochs

train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss=loss_function,
    optim=optimizer,
    schedule=scheduler,
    max_epochs=600,
    model_dir= config['paths']['model_dir'],
    device=device,
    val_interval=1
)

