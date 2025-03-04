import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH):
    return A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.2),
        ToTensorV2()
    ])

def get_val_transforms(IMAGE_HEIGHT, IMAGE_WIDTH):
    return A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2()
    ])