import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class Dataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        """
        Args:
            image_dir (str): Path to the folder containing images.
            annotations (dict): A dictionary where keys are image filenames, 
                                 and values are dicts containing 'boxes' and 'labels'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.image_fps = list(annotations.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_fps)

    def __getitem__(self, idx):
        img_name = self.image_fps[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Get annotations for this specific image
        anno = self.annotations[img_name]
        
        # Convert bounding boxes and labels to PyTorch Tensors
        # Format expected by Faster R-CNN: [xmin, ymin, xmax, ymax]
        boxes = torch.as_tensor(anno['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(anno['labels'], dtype=torch.int64)
        
        # Wrap annotations in a target dictionary
        target = {
            "boxes": boxes,
            "labels": labels
        }

        # Apply transforms (Data Augmentation / ToTensor)
        if self.transform:
            # Note: If using Albumentations for bounding boxes, 
            # the syntax changes slightly to pass both image and bboxes
            image = self.transform(image)
            
        return image, target