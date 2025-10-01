import os
import csv

# Define the root directory containing the age-specific folders
root_dir = 'data'

# Open the CSV file in write mode
with open('image_labels.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['file_path', 'label'])  # Write the header row

    # Iterate over each age group folder
    for age_group in os.listdir(root_dir):
        age_group_path = os.path.join(root_dir, age_group)

        # Ensure the path is a directory
        if os.path.isdir(age_group_path):
            # Iterate over each image file in the age group folder
            for image_name in os.listdir(age_group_path):
                image_path = os.path.join(age_group_path, image_name)

                # Ensure it's a file (not a subdirectory)
                if os.path.isfile(image_path):
                    # Write the image path and label (age group) to the CSV
                    writer.writerow([image_path, age_group])
