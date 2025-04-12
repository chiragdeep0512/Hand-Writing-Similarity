import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from itertools import combinations
from torch.nn.functional import pairwise_distance
from siamese_model import SiameseNetwork

# ===== Load the model =====
model = SiameseNetwork()
model.load_state_dict(torch.load("siamese_model.pth", map_location=torch.device('cpu')))
model.eval()

# ===== Image folder path =====
image_folder = "dataset/images"

# ===== Transformation (same as training) =====
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

# ===== Get all image filenames (e.g., w001_0.png, etc.) =====
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# ===== Generate all pair combinations (can limit to a few for demo) =====
image_pairs = list(combinations(image_files, 2))[:10]  # Show first 10 pairs (you can increase)

# ===== Loop through each image pair and show =====
for img1_name, img2_name in image_pairs:
    # Load and transform images
    img1_path = os.path.join(image_folder, img1_name)
    img2_path = os.path.join(image_folder, img2_name)

    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')

    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)

    # Get model output
    with torch.no_grad():
        out1, out2 = model(img1_tensor, img2_tensor)
        distance = pairwise_distance(out1, out2)
        similarity_score = (1 - distance.item()) * 100

    # Show the images and similarity score
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title(f"{img1_name}")
    ax[0].axis('off')

    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title(f"{img2_name}")
    ax[1].axis('off')

    plt.suptitle(f"Similarity Score: {similarity_score:.2f} %", fontsize=14)
    plt.tight_layout()
    plt.show()

    input("Press Enter to see next pair...\n")
