#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
# Importing the modules
import os
import shutil
import sys
import csv

param_mockups_path = sys.argv[1] if len(sys.argv) > 1 else "media\\screenshots\\mockups\\"
param_screenshots_path = sys.argv[2] if len(sys.argv) > 2 else "media\\screenshots\\"
param_iterations = sys.argv[3] if len(sys.argv) > 3 else 10
param_sequence = sys.argv[4] if len(sys.argv) > 4 else [1,1,2,2,1,1,1,2,2,2]

image_names = os.listdir(param_mockups_path)

images_dict = {1: [], 2: []}

for image in image_names:
    if "V1" in image:
        images_dict[1].append(image)
    elif "V2" in image:
        images_dict[2].append(image)

print(images_dict)

counter = 1
image_names_csv = []


for i in range(param_iterations):
    for j in param_sequence:
        for index, img in enumerate(images_dict[j]):
            # name = "image"+str(i)+str(index)+".png"
            name = "image"+str(counter)+".png"
            image_names_csv.append([name])
            src_file = os.path.join(os.getcwd(), param_mockups_path + img)
            dst_file = os.path.join(os.getcwd(), param_screenshots_path + name)
            # print(img + ", " + src_file + ", " + dst_file)
            shutil.copy(src_file,dst_file) #copy the file to destination dir
            counter = counter + 1

with open(param_screenshots_path + 'image_names.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(["Image_names"])
      wr.writerows(image_names_csv)
myfile.close()