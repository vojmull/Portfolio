import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import PUMADataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import logging

EPSILON = 1e-8

def save_checkpoint(model, filename='checkpoint.pth.tar'):
    logging.info('=> Saving Checkpoint')
    torch.save(model.state_dict(), filename)

def load_checkpoint(checkpoint, model):
    logging.info('=> Loading Checkpoint')
    model.load_state_dict(checkpoint)

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    class_mapping,
    num_workers=4,
    pin_memory=True,
):
    train_ds = PUMADataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PUMADataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda", class_mapping=None):
    if class_mapping is None:
        raise ValueError("class_mapping must be provided")

    num_correct = 0
    num_pixels = 0
    dice_score_total = 0.0
    model.eval()

    dice_per_class = [0 for i in range(len(class_mapping))]
    class_count = [0 for i in range(len(class_mapping))]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.argmax(preds, dim=1)

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            dice_score = 0.0
            for cls in range(len(class_mapping)):
                pred_cls = (preds == cls).float()
                target_cls = (y == cls).float()
                intersection = (pred_cls * target_cls).sum().item()
                dice_class = (2 * intersection + EPSILON) / (pred_cls.sum().item() + target_cls.sum().item() + EPSILON)
                dice_per_class[cls] += dice_class
                class_count[cls] += 1
                dice_score += dice_class
            dice_score_total += dice_score / len(class_mapping)

    accuracy = num_correct / num_pixels if num_pixels > 0 else 0
    average_dice_score = dice_score_total / len(loader) if len(loader) > 0 else 0

    for cls in range(len(class_mapping)):
        dice_per_class[cls] /= class_count[cls]

    logging.info(f"Accuracy: {accuracy*100:.3f}%")
    logging.info(f"Average Dice Score: {average_dice_score:.3f}")
    for cls in range(len(class_mapping)):
        logging.info(f"Dice per class {cls}: {dice_per_class[cls]:.3f}")
    return accuracy, average_dice_score


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", color_mapping=None):
    model.eval()

    def apply_color_palette(mask):
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in color_mapping.items():
            color_mask[mask == class_idx] = np.array(color, dtype=np.uint8)
        return color_mask

    os.makedirs(folder, exist_ok=True)
    logging.info(f'=> Saving predictions as imgs to {folder}')
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

        for i in range(preds.shape[0]):
            orig_img = Image.fromarray((x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            gt_np = y[i].cpu().numpy().astype(np.uint8)

            pred_color = apply_color_palette(pred_np)
            gt_color = apply_color_palette(gt_np)

            pred_img = Image.fromarray(pred_color)
            gt_img = Image.fromarray(gt_color)

            combined_width = orig_img.width + pred_img.width + gt_img.width
            combined_height = max(orig_img.height, pred_img.height, gt_img.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))
            combined_img.paste(orig_img, (0, 0))
            combined_img.paste(gt_img, (orig_img.width, 0))
            combined_img.paste(pred_img, (orig_img.width + gt_img.width, 0))

            combined_img.save(os.path.join(folder, f"combined_{idx}_{i}.png"))

        logging.debug(f"Saved {preds.shape[0]} images")

def calculate_class_weights(loader, num_classes, device="cuda"):
    # class_counts = torch.zeros(num_classes, device=device)
    # total_pixels = 0
    
    # for _, mask in loader:
    #     mask = mask.to(device)
    #     for class_id in range(num_classes):
    #         class_counts[class_id] += (mask == class_id).sum()
    #     total_pixels += mask.numel()
    
    # class_weights = total_pixels / (class_counts * num_classes)
    class_weights = torch.tensor([1.0, 2.0, 2.0, 4.0, 8.0, 8.0], device=device)
    logging.info(f"Class weights: {class_weights}")
    return class_weights    

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.5, epsilon=EPSILON, region_weighting=True):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.region_weighting = region_weighting
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.class_weights = class_weights

    def forward(self, preds, targets):
        # Cross-Entropy Loss
        ce_loss = self.cross_entropy_loss(preds, targets)

        # Dice Loss
        preds = torch.softmax(preds, dim=1)
        num_classes = preds.shape[1]
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice_losses = []
        for c in range(num_classes):
            pred_c = preds[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            pred_sum = pred_c.sum(dim=(1, 2))
            target_sum = target_c.sum(dim=(1, 2))

            dice_score = (2. * intersection + self.epsilon) / (pred_sum + target_sum + self.epsilon)
            dice_loss = 1.0 - dice_score

            if self.region_weighting:
                region_size = target_c.sum(dim=(1, 2))
                region_weight = 1.0 + region_size / (targets_one_hot.sum(dim=(2, 3)).max() + self.epsilon)
                dice_loss = dice_loss * region_weight

            if self.class_weights is not None:
                dice_loss = dice_loss * self.class_weights[c]

            dice_losses.append(dice_loss.mean())

        total_dice_loss = torch.stack(dice_losses).mean()

        combined_loss = self.alpha * ce_loss + (1 - self.alpha) * total_dice_loss
        return combined_loss
