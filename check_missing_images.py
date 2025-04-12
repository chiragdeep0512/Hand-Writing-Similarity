import os
import pandas as pd

# Change this to your actual image folder path
image_folder = r"D:\Hand wtiting project\dataset\images"
csv_path = r"D:\Hand wtiting project\dataset\labels.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Track missing images
missing_files = []

for img_name in df['image']:
    img_path = os.path.join(image_folder, img_name)
    if not os.path.exists(img_path):
        missing_files.append(img_name)

# Print summary
if missing_files:
    print(f"❌ Missing {len(missing_files)} images:")
    for img in missing_files:
        print(f"- {img}")
else:
    print("✅ All images in labels.csv are present.")

# Optional: Save missing image names to a CSV
if missing_files:
    pd.DataFrame(missing_files, columns=["missing_images"]).to_csv("missing_images.csv", index=False)
    print("📄 Saved missing image list to 'missing_images.csv'")
