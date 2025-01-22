import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from data.imageio import readImage,readLabel

class Data_Iterator(keras.utils.Sequence) :
    def __init__(self, image_dir, image_files, label_dir, label_files, batch_size,label_classes=6) :
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = image_files
        self.labels = label_files
        self.batch_size = batch_size
        self.label_classes = label_classes
      
    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        # Note: apparently in Python we do not have to take care if end index is larger than available?
        imbatch = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        labbatch = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        return (np.array([readImage(self.image_dir + '/' + img_name) for img_name in imbatch]),
                to_categorical(np.array([readLabel(self.label_dir + '/' + lab_name) for lab_name in labbatch]),self.label_classes))


class Predict_Iterator(keras.utils.Sequence) :
    def __init__(self, image_dir, image_files, batch_size) :
        self.image_dir = image_dir
        self.images = image_files
        self.batch_size = batch_size
      
    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        # Note: apparently in Python we do not have to take care if end index is larger than available?
        imbatch = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        
        return np.array([readImage(self.image_dir + '/' + img_name) for img_name in imbatch])


