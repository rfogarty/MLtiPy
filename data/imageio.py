import cv2 as cv
import numpy as np
import os
import random
#from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer
from paths import *

def imageName2LabelName(image_name) :
    # TODO: this is pretty junky code and not at all generalizable
    img_id = image_name.split('t')[0]
    img_suffix = image_name.split('t')[1]
    label_name = img_id + 'mask_t' + img_suffix
    return label_name

def readImage(impath,img_rows=320,img_cols=320,img_channels=1,normalize=True) :
    image = cv.imread(impath)
    if (img_channels == 1) : image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    elif (img_channels == 3) : image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    #image = resize(image,(img_rows,img_cols,img_channels),preserve_range=False)
    image = cv.resize(image, (img_rows, img_cols))
    
    if normalize :
        image = image.astype('float32')
        m=np.mean(image)
        s=np.std(image)
        image = (image - m)/s
    return image

def readLabel(labpath,img_rows=320,img_cols=320,img_channels=1) :
    label = cv.imread(labpath,cv.IMREAD_GRAYSCALE)
    label = resize(label, (img_rows,img_cols,img_channels),preserve_range = True)
    return label


def getImagesAndLabels(image_dir) :
    #_,_,image_files = os.walk(image_dir).__next__()
    image_files = list_images(image_dir)
    label_files = [imageName2LabelName(image_name) for image_name in image_files]
    return (image_files,label_files)

def readDataAndLabels(imagePaths,img_rows,img_cols,img_channels=1,data_dir='',normalize=True) :
    data=[]
    labels=[]
    paths=[]
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        imageFullPath = data_dir + imagePath
        paths.append(imageFullPath)
        #print(f'Opening: {imageFullPath}')
        #print(type(imageFullPath))
        image = readImage(imageFullPath,img_rows,img_cols,img_channels,normalize)
        #image = cv2.imread(imageFullPath)
        #print(image.shape)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (img_rows, img_cols))
        #image= image.astype('float32')
        #image =  cv2.normalize(image,
        #            None,
        #            alpha=0,
        #            beta=1,
        #            norm_type=cv2.NORM_MINMAX)
    
        #image = img_to_array(image)
        # Now standardize images to zero-mean and unit variance
        #m = np.mean(image)
        #s = np.std(image)
        #image = (image - m)/s
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    
    data = np.array(data)
    
    labels = np.array(labels)
    # binarize the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    print(f"Read {len(data)} Data Files.")

    return (data,labels,paths)

def getDataFromDir(data_dir,input_tensor_shape=(224,224,3),shuffled=True,filterstr=None,blocklist=None) :
    img_rows, img_cols, img_channel = input_tensor_shape
    
    # grab the image paths and randomly shuffle them
    print(f"[INFO] loading images from {data_dir}...")
    imagePaths = sorted(list(paths.list_images(data_dir)))
    filelist = filterFilelist(imagePaths,filterstr,blocklist)
    if shuffled:
        random.seed(14)
        random.shuffle(imagePaths)
    
    return readDataAndLabels(imagePaths,img_rows,img_cols)


def getDataFromListing(data_dir,file_list,input_tensor_shape=(224,224,3),shuffled=True,filter=None,blocklist=None) :
    img_rows, img_cols, img_channel = input_tensor_shape
    data=[]
    labels= []
    
    # grab the image paths and randomly shuffle them
    print(f"[INFO] loading images from {file_list}...")
    imagePaths = None
    with open(os.path.join(data_dir,file_list)) as f:
        lines = [line.strip() for line in f.readlines()]
        imagePaths=sorted(lines)
    filelist = filterFilelist(imagePaths,filterstr,blocklist)
    if shuffled:
        random.seed(14)
        random.shuffle(imagePaths)
    
    return readDataAndLabels(imagePaths,img_rows,img_cols,data_dir + os.path.sep)


