import os

images_dir_path = ('../data/DRLung/test/original')
labels_dir_path = ('../data/DRLung/test/mask')

images_file_name = [i for i in os.listdir(images_dir_path) if i[-3:] == 'png']
images_path = [os.path.join(images_dir_path, i) for i in images_file_name]
f = open("../data/DRLung/split/test_images.txt", "w")
for line in images_path:
    f.write(line + '\n')
f.close()

labels_path = [os.path.join(labels_dir_path, i) for i in images_file_name]
f = open("../data/DRLung/split/test_labels.txt", "w")
for line in labels_path:
    f.write(line + '\n')
f.close()
