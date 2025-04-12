import pandas as pd

# Paths
csv_path = r"D:\Hand wtiting project\dataset\labels.csv"
missing_images_path = "missing_images.csv"
filtered_csv_path = r"D:\Hand wtiting project\dataset\labels_cleaned.csv"

# Load files
df = pd.read_csv(csv_path)
missing = pd.read_csv(missing_images_path)

# Filter out rows with missing images
df_clean = df[~df['image'].isin(missing['missing_images'])]

# Save cleaned CSV
df_clean.to_csv(filtered_csv_path, index=False)
print(f"✅ Cleaned CSV saved as: {filtered_csv_path} with {len(df_clean)} entries")

import pandas as pd

df = pd.read_csv(r"D:\Hand wtiting project\dataset\labels_cleaned.csv")
print("Remaining rows after filtering:", len(df))
print(df.head())
