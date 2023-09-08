from rembg import remove
from PIL import Image
if __name__ == '__main__':
    image = Image.open("D:\\323072238_705624701226124_3940280021750688045_n.jpg")
    image = remove(image)
    image.show()