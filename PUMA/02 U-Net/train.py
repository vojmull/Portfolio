import logging
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import wandb

from augmentations import get_train_transforms, get_val_transforms
from early_stopping import EarlyStopping
from model import UNETWithAttention
from utils import (CombinedLoss, calculate_class_weights, load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# HYPERPARAMETERS GROUP A
IN_CHANNELS = 3
TRAIN_IMG_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"
VAL_IMG_DIR = "data/test/images"
VAL_MASK_DIR = "data/test/masks"
IMAGE_HEIGHT = 512 if DEVICE == "cuda" else 64
IMAGE_WIDTH = 512 if DEVICE == "cuda" else 64
CLASS_MAPPING = {
    'tissue_tumor': 3,
    'tissue_stroma': 1,
    'tissue_blood_vessel': 2,
    'tissue_epidermis': 4,
    'tissue_white_background': 0,
    'tissue_necrosis': 5
}
COLOR_MAPPING = {
    3: [200, 0, 0],
    1: [150, 200, 150],
    2: [0, 255, 0],
    4: [99, 145, 164],
    0: [255, 255, 255],
    5: [51, 0, 51]
}
OUT_CHANNELS = len(CLASS_MAPPING)

# HYPERPARAMETERS GROUP B
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 1000
NUM_WORKERS = 1
PATIENCE = 100
PIN_MEMORY = True
LOAD_MODEL = False
A_MEAN = [0., 0., 0.]
A_STD = [1., 1., 1.]

def train_fn(train_loader, model, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def validate_fn(val_loader, model, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    train_transforms = get_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    val_transforms = get_val_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)

    model = UNETWithAttention(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    logging.info(f"Training on: {DEVICE}")

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, 
        BATCH_SIZE, train_transforms, val_transforms, CLASS_MAPPING, 
        NUM_WORKERS, PIN_MEMORY
    )

    class_weights = calculate_class_weights(train_loader, num_classes=OUT_CHANNELS, device=DEVICE)
    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint.pth.tar"), model)

    loss_fn = CombinedLoss(class_weights=class_weights, alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss = validate_fn(val_loader, model, loss_fn, DEVICE)
        scheduler.step()
        acc, dice = check_accuracy(val_loader, model, device=DEVICE, class_mapping=CLASS_MAPPING)
        logging.info(f"Epoch {epoch+1} Val loss: {val_loss:.3f}, Val accuracy: {acc:.3f}, Val dice score: {dice:.3f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Stopping training.")
            save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE, color_mapping=COLOR_MAPPING)
            load_checkpoint(torch.load("checkpoint.pth.tar"), model)
            break

        if epoch % 25 == 0:
            save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE, color_mapping=COLOR_MAPPING)
            imgs = [Image.open(os.path.join("saved_images/", img)) for img in os.listdir("saved_images/") if img.endswith(".png")]
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": acc,
                "val_dice_score": dice,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "images": [wandb.Image(img) for img in imgs[:30]]
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": acc,
                "val_dice_score": dice,
                "learning_rate": optimizer.param_groups[0]["lr"]
            })

        if early_stopping.has_improved(val_loss):
            save_checkpoint(model)

if __name__ == '__main__':
    wandb.init(project="U-Net", config={
        "name": "U-Net_15 Npy masks",
        "notes": "Changed geojson to npy masks",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "device": DEVICE,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "mean": A_MEAN,
        "std": A_STD
    })
    main()
    wandb.finish()
