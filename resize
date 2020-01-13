from PIL import Image
import os, sys

path = "C:\\Users\mauri\OneDrive\Documenten\project-YURA\plant_leaf_dataset\plant_leaf_dataset\Pepperbell_bacterial_spot"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        print(item)
        place = os.path.join(path,item)
        im = Image.open(place)
        print("a")
        f, e = os.path.splitext(path + item)
        imResize = im.resize((32, 32), Image.ANTIALIAS)
        print(imResize)
        imResize.save(place, 'JPEG', quality=90)


resize()
