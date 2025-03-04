import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PUMADataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.tif', '_tissue.npy'))
        image = np.array(Image.open(img_path).convert('RGB'))
        image = image.astype(np.float32) / 255.0
        mask = np.load(mask_path).astype(np.uint8)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask'].long()
        
        return image, mask