# import csv

# # Open the existing CSV file in read mode
# with open("test.csv", "r") as existing_file:
#     reader = csv.reader(existing_file)
#     existing_data = list(reader)

# # Open the new CSV file in read mode
# with open("white_ratio6-1.csv", "r") as new_file:
#     reader = csv.reader(new_file)
#     new_data = list(reader)[1:]  # Skip the header

# # Create a list to store the merged data
# merged_data = []

# # Add the data from the existing CSV file
# for row in existing_data:
#     merged_data.append(row)

# # Add the new data
# for row in new_data:
#     # Find the image name in the existing data
#     for merged_row in merged_data:
#         if merged_row[0] == row[0]:
#             # Append the white ratio to the end of the row
#             merged_row.append(row[1])
#             break
#     else:
#         # If the image name was not found, print an error and stop execution
#         print(f"Error: The image {row[0]} was not found in the existing data.")
#         exit(1)

# # Write the merged data to a new CSV file
# with open("glare_true_pred6-1.csv", "w", newline="") as merged_file:
#     writer = csv.writer(merged_file)
#     writer.writerows(merged_data)

import csv

# Open the existing CSV file in read mode
with open("test.csv", "r") as existing_file:
    reader = csv.reader(existing_file)
    existing_data = list(reader)

# Open the new CSV file in read mode
with open("white_ratio6-1.csv", "r") as new_file:
    reader = csv.reader(new_file)
    new_data = list(reader)[1:]  # Skip the header

# Add new column title to the existing header
existing_data[0].append("White Pixel Ratio")

# Add the new data
for row in new_data:
    # Find the image name in the existing data
    for existing_row in existing_data[1:]:  # Skip the header
        if existing_row[0] == row[0]:
            # Append the white ratio to the end of the row
            existing_row.append(row[1])
            break
    else:
        # If the image name was not found, print an error and stop execution
        print(f"Error: The image {row[0]} was not found in the existing data.")
        exit(1)

# Write the merged data to a new CSV file
with open("glare_ture_pred6-1.csv", "w", newline="") as merged_file:
    writer = csv.writer(merged_file)
    writer.writerows(existing_data)
