import os
import pandas as pd

# Path to your image directory
image_dir = r"D:\Hand wtiting project\dataset\images"
# Path to original CSV
original_csv = r"D:\Hand wtiting project\dataset\labels.csv"
# Output path
cleaned_csv = r"D:\Hand wtiting project\dataset\labels_cleaned.csv"

# Load original CSV
df = pd.read_csv(original_csv)

# Get list of image filenames in folder
existing_images = set(os.listdir(image_dir))

# Keep only rows where image exists
df_cleaned = df[df['image'].isin(existing_images)]

# Save cleaned CSV
df_cleaned.to_csv(cleaned_csv, index=False)
print(f"✅ Cleaned CSV saved to: {cleaned_csv} with {len(df_cleaned)} valid rows.")

import os

image_dir = r"D:\Hand wtiting project\dataset\images"
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

import os

image_dir = "D:/Hand wtiting project/dataset/images"
images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

print(f"✅ Found {len(images)} PNG images.")
