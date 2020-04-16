# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append('/home/listic/Downloads/Maritme_Mask_RCNN Pano/mrcnn')  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import Maritime config
sys.path.append(os.path.join(ROOT_DIR, "/home/listic/Downloads/Maritme_Mask_RCNN Pano/mrcnn"))  # To find local version
#from pycocotools.coco import COCO
import maritime

from matplotlib.pyplot import imshow
#get_ipython().magic(u'matplotlib inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mrcnn/logs/maritime20200409T1730/mask_rcnn_maritime_0182.h5")


# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "test_images/")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(maritime.MaritimeConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
#class_names = ['undetected','porte','luminaire','prise','interrupteur','radiateur','extincteur','fenetre','controleur','tetes d edi']
#colors= [[0,0,0],[0,255,0],[128,0,0],[255,255,255],[0,255,255],[255,255,128],[128,255,255],[128,128,255],[255,255,192],[192,192,255]]

class_names = ['undetected','porte','feneter','lampe','prise','interrupteur','radiateur','baes', 'extincteur','controleur']
colors= [[0,0,0],[0,255,0],[128,0,0],[255,255,255],[0,255,255],[255,255,128],[128,255,255],[128,128,255],[255,255,192],[192,192,255]]


# ## Run Object Detection

# In[46]:



for filename in os.listdir(IMAGE_DIR):
# Load a random image from the images folder
    #file_names = next(os.walk(IMAGE_DIR))[2]
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR,(filename)))

    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results 
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    #visualize.save_image(image, filename+'r', r['rois'], r['masks'], r['class_ids'],r['scores'], class_names)
#print(r['class_ids'], class_names)

# In[26]:

