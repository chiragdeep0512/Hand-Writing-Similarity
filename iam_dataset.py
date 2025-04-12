# iam_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((105, 105)),
            transforms.ToTensor()
        ])

        self.image_paths = []
        self.labels = []
        self.label_to_images = {}

        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    label = os.path.basename(root)
                    path = os.path.join(root, file)
                    self.image_paths.append(path)
                    self.labels.append(label)

                    if label not in self.label_to_images:
                        self.label_to_images[label] = []
                    self.label_to_images[label].append(path)

    def __getitem__(self, index):
        # Get first image and its label
        img1_path = self.image_paths[index]
        label1 = self.labels[index]
        img1 = Image.open(img1_path).convert("L")

        # Decide if the pair is similar or dissimilar
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            img2_path = random.choice(self.label_to_images[label1])
        else:
            label2 = random.choice([lbl for lbl in self.label_to_images if lbl != label1])
            img2_path = random.choice(self.label_to_images[label2])

        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = 1 if should_get_same_class else 0
        return img1, img2, label

    def __len__(self):
        return len(self.image_paths)
