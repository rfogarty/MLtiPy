
import matplotlib.pyplot as plt
import numpy as np

def plotMetric(plotfilename,metric,H,N) :
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.plot(np.arange(0, N), H.history[metric], label="train_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel(metric)
    plt.legend(['train'], loc='upper left')
    plt.savefig(plotfilename)

