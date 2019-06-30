from PIL import Image
import numpy as np

def resize(screen, shape=(250, 150)):
    im = Image.fromarray(screen, mode='RGB')
    im = im.resize(size=shape, resample=Image.BILINEAR) # size = (width, height)
    im = np.asarray(im)
    
    return im/255.0