# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:39:11 2025

@author: panos
"""

import os
import shutil
import pandas as pd


# Path to the unified images folder
image_folder = "./airplanes"

# List to store image names and labels
data = []

# Loop through all JPG files
for img_name in os.listdir(image_folder):
    if img_name.endswith(".jpg"):  # Process only JPG images
        label = img_name.split("_")[0]  # Extract label from filename prefix
        data.append([img_name, label])

# Convert list to DataFrame
df = pd.DataFrame(data, columns=["image_name", "label"])

# Save to CSV
csv_path = os.path.join('./', "image_labels.csv")
df.to_csv(csv_path, index=False)

print(f"CSV file created: {csv_path}")

