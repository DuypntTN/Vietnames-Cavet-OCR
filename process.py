import os
dir_images = './run/temp/'
dir_results = './run/results/'

# read folder names in dir_images
folders = os.listdir(dir_images)
print(folders)
for folder in folders:
    # read image in folder
    image = os.listdir(dir_images + folder + '/cavet_detector')
    print(image)
    # write image to dir_results
    os.rename(dir_images + folder + '/cavet_detector/' + image[0], dir_results + image[0])