import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from PIL import Image
import numpy as np

# Fix error: module 'numpy' has no attribute 'bool' when using numpy 1.24.0+
np.bool = np.bool_

# Fix error: module 'PIL.Image' has no attribute 'ANTIALIAS'
Image.ANTIALIAS = Image.LANCZOS


config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'

detector = Predictor(config)

img = './sample/031189003299.jpeg'
img = Image.open(img)
plt.imshow(img)
plt.show()
s = detector.predict(img)

print(s)