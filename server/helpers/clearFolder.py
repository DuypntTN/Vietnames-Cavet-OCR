import os



def clearFolderContent(dir):
    for file in os.listdir(dir):
        os.remove(os.path.join(dir, file))