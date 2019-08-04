from PIL import Image
import numpy as np

def load_image(img_name, resize=(256,256), expand=True):
    #Load the image with PIL Image
    img = Image.open(img_name)
    
	#If resize is not false, resize to given shape
    if resize != False:
        img = img.resize(resize, Image.LANCZOS)
    img = np.array(img)
	
	#Expand means expand front dimension to add a batch size of 1.
    if expand:
        img = np.expand_dims(img,0)
    
	#Return as np.float32 format
    return img.astype(np.float32)
	
def gray_to_rgb(img):
    rgb_temp = np.array([img,img,img]) #Greyscale value repeated for RGB channels
    return np.rollaxis(rgb_temp, 0, 3) #Transform from C,W,H to W,H,C
