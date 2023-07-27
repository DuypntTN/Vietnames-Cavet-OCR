from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import cv2


def processInputImage(im=None):
    im = cv2.resize(im, (0, 0), fx=4, fy=4)
    im_cp = im.copy()
    # Convert to grayscale:
    im_cp = cv2.cvtColor(im_cp, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(im_cp)
    return equalized


def process(image_path):
    # Open the image using PIL:
    im = Image.open(image_path)

    # Resize the image:
    im = im.resize((im.width * 4, im.height * 4))

    # Create a copy of the resized image:
    im_cp = im.copy()

    # Convert the image to grayscale:
    im_cp = ImageOps.grayscale(im_cp)

    # Moprhological operations to remove noise:
    im_cp = im_cp.filter(ImageFilter.MinFilter(3))
    im_cp = im_cp.filter(ImageFilter.MaxFilter(5))
    

    # Equalize the histogram:
    equalized = ImageOps.equalize(im_cp)

    # Show the equalized image:
    # equalized.show()

    return equalized
def processImageBeforeRecognitionText(image_path):

    im = Image.open(image_path)


    # Bicubic interpolation:
    im = im.resize((im.width * 8, im.height * 8), Image.BICUBIC)

    # # Histogram equalization:
    # im = ImageOps.equalize(im)

    # # Remove noise using gaussian filter:
    # im = im.filter(ImageFilter.GaussianBlur(1))

    # Convert to grayscale:
    im = ImageOps.grayscale(im)


    # # Show the image:
    # im.show()

    return im