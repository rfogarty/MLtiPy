from tensorflow.python.keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    #y_pred = y_pred.astype('float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


