
import os
from PIL import Image
import numpy as np
import csv

def get_subfolders(path):
    """
    获取指定路径下的所有子文件夹目录
    """
    subfolders = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return subfolders

def calculate_mean_squared_distances(total_path, total_path_register): #读取此文件夹，获取此文件夹中的所有子文件夹目录

    subfolders = os.listdir(total_path)
    subfolders_register = get_subfolders(total_path_register)

    print("Calculating mean squared distances for images in the following subfolders:")
    coutnumber=0
    for i in range(len(subfolders)):
        subfolder = os.path.join(total_path, subfolders[i])
        for j in range(len(subfolders)):
            if i!=j:
                 subfolder_register = os.path.join(total_path, subfolders[j])
                 print(subfolder_register)
                 print(subfolder)
                 img1 = Image.open(subfolder).convert('L')
                 img2 = Image.open(subfolder_register).convert('L')
                 img1 = np.array(img1, dtype=np.float32)
                 img2 = np.array(img2, dtype=np.float32)
                 mse = np.mean((img1 - img2) ** 2)
                 result_file = os.path.join(total_path, "mean_squared_distances_v44.csv")
                 with open(result_file, mode='a', newline='') as csvfile:
                      csv_writer = csv.writer(csvfile)
                      csv_writer.writerow([f"Mean squared distances for image {subfolders[i]} to {subfolders[j]} : ", mse])
        #entries_fix= os.listdir(subfolders_register[coutnumber])
        #search_string=   "im_fix"
        #search_stringfix =  "im_out_"
        #files_with_string = [entry for entry in entries]
        #files_with_stringfix = [entry for entry in entries]
        #print(files_with_string[0])
        #print(files_with_stringfix[0])

        #pathname = os.path.join(total_path,files_with_string[0])        
        #pathnamefix = os.path.join(total_path,files_with_stringfix[0])
        #print(pathname)
        #print(pathnamefix)
        #img1 = Image.open(pathname).convert('L')
        #img2 = Image.open(pathnamefix).convert('L')
        #img1 = np.array(img1, dtype=np.float32)
        #img2 = np.array(img2, dtype=np.float32)
        #mask=(img1!=0) & (img2!=0)
        #mse = np.mean((img1[mask] - img2[mask]) ** 2)

        #mse = np.mean((img1 - img2) ** 2)

        #result_file = os.path.join(total_path, "mean_squared_distances_v44.csv")
        #with open(result_file, mode='a', newline='') as csvfile:
                    #csv_writer = csv.writer(csvfile)
                    #csv_writer.writerow([f"Mean squared distances for image {files_with_stringfix[0]} : ", mse])
        #print(f"Mean squared distances for image {files_with_string[0]}  to {files_with_stringfix[0]}:", mse)
        #coutnumber=coutnumber+1
