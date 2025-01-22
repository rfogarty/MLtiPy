#import os
#import numpy as np
#import random
#import cv2
#from imutils import paths
#from keras.preprocessing.image import img_to_array


#def getTrainingData() :
#    #train_data_dir = '/home/r/rfogarty/train-sob5'
#    #train_data_dir = '/data/rfogarty/CovidTest/train-sob5'
#    #train_data_dir = '/raid/rfogarty/LabeledData/Train/Regions'
#    #train_data_dir = '/data/UnivMiamiProstatePathology/LabeledDataTight/Train/Regions'
#    #train_data_dir = '/data/UnivMiamiProstatePathology/LabeledDataTight/Train/Regions'
#    train_data_dir = '/data/UnivMiamiProstatePathology/LabeledDataTight/Train/RegionsAugmentedEvenSplit'
#    #img_rows, img_cols, img_channel = 224,224, 3
#    img_rows, img_cols, img_channel = 300,300, 3
#    input_tensor_shape=(img_rows, img_cols, img_channel)
#    #BS = 10
#    #BS = 80 # 6.5 s per epoch
#    #BS = 240 # ?s per epoch
#    BS = 120 # ?s per epoch
#    data=[]
#    labels= []
#    
#    # grab the image paths and randomly shuffle them
#    print("[INFO] loading images...")
#    imagePaths = sorted(list(paths.list_images(train_data_dir)))
#    random.seed(14)
#    random.shuffle(imagePaths)
#    # loop over the input images
#    for imagePath in imagePaths:
#        # load the image, pre-process it, and store it in the data list
#        image = cv2.imread(imagePath)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        image = cv2.resize(image, (img_rows, img_rows))
#        #image= image.astype('float32')
#        #image =  cv2.normalize(image,
#        #            None,
#        #            alpha=0,
#        #            beta=1,
#        #            norm_type=cv2.NORM_MINMAX)
#    
#        image = img_to_array(image)
#        # Now standardize images to zero-mean and unit variance
#        m = np.mean(image)
#        s = np.std(image)
#        image = (image - m)/s
#        data.append(image)
#        label = imagePath.split(os.path.sep)[-2]
#        labels.append(label)
#    
#    data = np.array(data)
#    
#    ## Now normalize histogram by removing the mean and unit normalizing stddev
#    #m=np.mean(data)
#    #s=np.std(data)
#    #data = (data - m)/s
#    
#    
#    labels = np.array(labels)
#    # binarize the labels
#    lb = LabelBinarizer()
#    labels = lb.fit_transform(labels)
#    print("Training data: ",len(data))
#    return (data,labels)


#datagen = ImageDataGenerator(
#          rotation_range=45, # rotation
#          width_shift_range=0.2, # horizontal shift
#          height_shift_range=0.2, # vertical shift
#          zoom_range=0.2, # zoom
#          horizontal_flip=True, # horizontal flip
#          vertical_flip=True, # vertical flip
#          brightness_range=[0.2,1.2]) # brightness

# Use the following form of model fit when using a data generator
#datagen_seed = 17
#H = model.fit(datagen.flow(data,labels,batch_size=BS,seed=datagen_seed),
#              epochs=nb_epoch,
#              callbacks=callbacks_list, verbose=1)


