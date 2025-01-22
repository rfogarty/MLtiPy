
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.initializers import glorot_uniform

#############################################################################################################
# Implementation note: much of the code in this file was adopted from Priya Dwivedi at
#    https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
#
# However, code has been modify by myself (Ryan B Fogarty) where new blocks were added and documented as such
# or single lines added and tagged with "RBF".
#############################################################################################################

# This function is courtesy of Priya Dwivedi as written here, and is one of the 2 common skip networks in a ResNet
#    https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
def identity_block(X, f, filters, stage, block,dropout=0.0):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X) ### RBF - added Dropout to layer

    
    # Second component of main path (≈3 lines)
    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X) ### RBF - added Dropout to layer

    # Third component of main path (≈2 lines)
    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X) ### RBF - added Dropout to layer

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    return X

# This function is courtesy of Priya Dwivedi as written here, and is one of the 2 common skip networks in a ResNet
#    https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
def convolutional_block(X, f, filters, stage, block, s = 2,dropout=0.0):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = layers.Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X)

    # Second component of main path (≈3 lines)
    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X) ### RBF - added Dropout to layer


    # Third component of main path (≈2 lines)
    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X) ### RBF - added Dropout to layer


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    if dropout > 0.0 : X_shortcut = layers.Dropout(dropout)(X_shortcut) ### RBF - added Dropout to layer

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    
    
    return X


# Mini-ResNet12 tailored for MNist classification, which is a heavily modified version of a ResNet50.
# The ResNet-50 was written by Priya Dwivedi, and also borrows from her identity and convolution blocks above.
#    https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
def buildResNet12(input_shape=(28, 28, 1), classes=10,dropout=0.0, preprocessing=None):
    """
    Implementation of the popular ResNet50 reduced to a ResNet12, with the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = layers.Input(input_shape)
    lastLayer = X_input
    
    ############################### Ryan B Fogarty #############################
    # This step added to Priya's code to allow preprocessing layers that add data augmentation during training
    # (note, we only want these enabled for the training network and not the inference network).
    if preprocessing != None :
        for layer in preprocessing :
            print(f"INFO: adding preprocessing stage {layer} to model {lastLayer}")
            lastLayer = layer(lastLayer)
    ######################### / Ryan B Fogarty #################################

    # Zero-Padding
    X = layers.ZeroPadding2D((3, 3))(lastLayer)

    # Stage 1
    ######################### Ryan B Fogarty ###################################
    # Input convolution layer kernel size significantly reduced over standard ResNet50 since MNist images are so small
    #X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.Conv2D(32, (3, 3), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    ######################### / Ryan B Fogarty #################################
    X = layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = layers.Activation('relu')(X)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X) ### RBF - added Dropout to stage
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1,dropout=dropout)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b',dropout=dropout)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c',dropout=dropout)
    if dropout > 0.0 : X = layers.Dropout(dropout)(X) ### RBF - added Dropout to stage

    ############################### Ryan B Fogarty #############################
    # The following stages were removed since we would like a model with much 
    #   fewer parameters vs. ResNet50 
    # (removed) Stage 3 (≈4 lines)
    # (removed) Stage 4 (≈6 lines)
    # (removed) Stage 5 (≈3 lines)
    ######################### / Ryan B Fogarty #################################

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = layers.AveragePooling2D((2,2), name="avg_pool")(X)

    # output layer
    X = layers.Flatten()(X)
    X = layers.Dense(64, activation='relu', name='hidden_features', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet12')
    model.summary()
    return model


# This function creates a training network with the "deeper" ResNet12 model that
# was derived from the larger ResNet50 model.
def createResNet12TrainingModel(seed=23) :
    model = buildResNet12(input_shape = (28, 28, 1), classes = 10,dropout=0.2,preprocessing=[
      layers.experimental.preprocessing.RandomRotation(0.2,seed=seed), 
      layers.experimental.preprocessing.RandomZoom(height_factor=(0.1, 0.2), width_factor=(0.1, 0.2),seed=seed)])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    
    return model


# This function creates an inference network for the ResNet12 so that is ready to evaluate on a test set
def createResNet12InferenceModel() :
    
    model = buildResNet12(input_shape = (28, 28, 1), classes = 10)

    return model


# This function creates an inference network which is an ensemble of ResNet12s trained
# with the "snapshot ensemble" technique. Due to the fact that this does not create
# a proper Keras or TensorFlow model, we can not use the typical model.evaluate
# technique to compute accuracy. Instead, we use functions in the predictMNist.py file
# are set up to evaluate and compute statistics.
def createResNet12EnsembleModel(models) :
    modelEnsemble = [0] * len(models)
    for idx,snapshot_epoch in enumerate(models) :
        filename=f"res3-weights2-improvement-{snapshot_epoch:03d}.hdf5"
        print(f'INFO: adding {filename} to ensemble')
        modelEnsemble[idx] = createDeeperInferenceModel()
        modelEnsemble[idx].build(input_shape = (1,28,28,1))
        modelEnsemble[idx].load_weights(filename)
    return modelEnsemble


def buildNet(input_tensor_shape,seed=17) :
    #strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3","GPU:4", "GPU:5", "GPU:6", "GPU:7"])
    #strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3", "GPU:4", "GPU:5"])
    #strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
    #strategy = tf.distribute.MirroredStrategy(["GPU:2", "GPU:3"])
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    
    with strategy.scope():
    
        inputModel = layers.Input(input_tensor_shape)
        headModel = inputModel
        headModel = layers.experimental.preprocessing.RandomFlip(seed=seed)(headModel)
        headModel = layers.experimental.preprocessing.RandomRotation(0.2,seed=seed)(headModel)
        headModel = layers.experimental.preprocessing.RandomZoom(height_factor=(0.1, 0.2), width_factor=(0.1, 0.2),seed=seed)(headModel)
        
        base_model = ResNet50(weights='imagenet', include_top=False)
        for layer in base_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable  = False
        
        headModel = base_model(headModel)
        headModel = GlobalAveragePooling2D()(headModel)
        #headModel = Dense(64, activation='relu')(headModel)
        headModel = Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                          bias_regularizer=regularizers.l2(1e-4))(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(1, activation='sigmoid')(headModel)
        model = Model(inputs=inputModel, outputs=headModel)
        
        print(model.summary())
        
        opt = SGD(momentum=0.9)
         
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.Recall(),
                               tf.keras.metrics.TruePositives(),
                               tf.keras.metrics.TrueNegatives(),
                               tf.keras.metrics.FalsePositives(),
                               tf.keras.metrics.FalseNegatives(),
                               tf.keras.metrics.AUC(num_thresholds=20)])
    return model

def buildInferenceNet(input_tensor_shape) :
    #strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1","GPU:2","GPU:3"])
    strategy = tf.distribute.OneDeviceStrategy(device="GPU:0") 
    with strategy.scope():
        inputModel = layers.Input(input_tensor_shape)
        headModel = inputModel
        
        base_model = ResNet50(weights='imagenet', include_top=False)
        for layer in base_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable  = False
        
        headModel = base_model(headModel)
        headModel = GlobalAveragePooling2D()(headModel)
        headModel = Dense(64, activation='relu')(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dense(1, activation='sigmoid')(headModel)
        model = Model(inputs=inputModel, outputs=headModel)
    
    return model


