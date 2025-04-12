import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

        # Build a mapping from writer -> list of image filenames
        self.writer_to_images = {}
        for _, row in self.data.iterrows():
            writer = row['writer']
            image = row['image']
            if writer not in self.writer_to_images:
                self.writer_to_images[writer] = []
            self.writer_to_images[writer].append(image)

        # All unique writer IDs
        self.writers = list(self.writer_to_images.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_row = self.data.iloc[idx]
        anchor_img_path = os.path.join(self.image_folder, anchor_row['image'])
        anchor_writer = anchor_row['writer']

        # Load anchor image
        anchor_img = Image.open(anchor_img_path).convert('L')

        # Decide whether to make a positive or negative pair
        is_positive = random.choice([True, False])

        if is_positive:
            # Choose another image with same writer
            possible_imgs = self.writer_to_images[anchor_writer]
            if len(possible_imgs) < 2:
                return self.__getitem__((idx + 1) % len(self))
            img2_name = random.choice([img for img in possible_imgs if img != anchor_row['image']])
            label = 1
        else:
            # Choose image from a different writer
            negative_writer = random.choice([w for w in self.writers if w != anchor_writer])
            img2_name = random.choice(self.writer_to_images[negative_writer])
            label = 0

        img2_path = os.path.join(self.image_folder, img2_name)
        img2 = Image.open(img2_path).convert('L')

        # Apply transform if available
        if self.transform:
            anchor_img = self.transform(anchor_img)
            img2 = self.transform(img2)

        return (anchor_img, img2), label
