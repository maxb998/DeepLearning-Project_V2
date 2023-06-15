import os
import shutil
from zipfile import ZipFile
from zipfile import is_zipfile

# Path alla directory che contiene tutti gli zip da spacchettare
directory = "./data"

# Boh in caso sia un'altra l'estensione delle label
label_extension = ".txt"

# Creazione directory per immagini e label se non esiste gia'
new_images_dir = os.path.join(directory, "images")
new_labels_dir = os.path.join(directory, "labels")
if not os.path.isdir(new_images_dir): 
    os.mkdir(new_images_dir)
    print("Created images directory")
if not os.path.isdir(new_labels_dir): 
    os.mkdir(new_labels_dir)
    print("Created labels directory")


old_images_dir = os.path.join(directory, "synthetic_dataset/images")
old_labels_dir = os.path.join(directory, "synthetic_dataset/labels")

# Per rinominare i file
count = 0

for zipname in sorted(filter(lambda x: is_zipfile(os.path.join(directory, x)), os.listdir(directory))):
    
    # Estrazione dello zip
    zf = ZipFile(os.path.join(directory, zipname))
    zf.extractall(directory)
    print(zipname + " extracted")
    zf.close()

    # Rinominazione e spostamento di immagini e label
    for file in sorted(os.listdir(old_images_dir)):
        
        splitted_name = file.split(".")
        label = splitted_name[0] + label_extension
        new_img_name = str(count).zfill(6) + "." + splitted_name[1]
        new_lbl_name = str(count).zfill(6) + label_extension

        os.rename(os.path.join(old_images_dir, file), os.path.join(new_images_dir, new_img_name))
        print("Moved " + file + " and renamed into " + new_img_name)

        os.rename(os.path.join(old_labels_dir, label), os.path.join(new_labels_dir, new_lbl_name))
        print("Moved " + label + " and renamed into " + new_lbl_name)

        count += 1
    print(zipname + " -> done!")
    shutil.rmtree(directory + "/synthetic_dataset")
    print("Removed the old synthetic_dataset directory")
