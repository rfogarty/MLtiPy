import pandas as pd
import numpy as np
import sys
from stats import *

np.set_printoptions(threshold=sys.maxsize)

logsAll,logsPar,logsAvg,logs=loadLogs()
altStats=derivedTrainStats(logsAll)
altValStats=derivedValStats(logsAll)
logsAll=pd.concat([logsAll,altStats,altValStats],axis=1)
boxplotStat('foldaccuracies.png',logsPar,['binary_accuracy'])
boxplotStat('foldvalaccuracies.png',logsPar,['val_binary_accuracy'])
boxplotStat('average_accuracies.png',logsAll,['binary_accuracy','val_binary_accuracy'],relabel=False)
#boxplotStat('medstats.png',logsAll,['specificity','sensitivity','auc','precision','dice','kappa'],relabel=False)
boxplotStat('medstats.png',logsAll,['specificity','sensitivity','auc','dice','kappa'],relabel=False)
#boxplotStat('medvalstats.png',logsAll,['val_specificity','val_sensitivity','val_auc','val_precision','val_dice','val_kappa'],relabel=False)
boxplotStat('medvalstats.png',logsAll,['val_specificity','val_sensitivity','val_auc','val_dice','val_kappa'],relabel=False)

lineplotStat('foldaccuracy.png',logsPar,['binary_accuracy'])
lineplotStat('foldvalaccuracy.png',logsPar,['val_binary_accuracy'])
lineplotStat('avgfoldaccuracy.png',logsAvg,['binary_accuracy'])
lineplotStat('avgfoldvalaccuracy.png',logsAvg,['val_binary_accuracy'])

lineplotStat('foldloss.png',logsPar,['loss'])
lineplotStat('foldvalloss.png',logsPar,['val_loss'])
lineplotStat('avgfoldloss.png',logsAvg,['loss'])
lineplotStat('avgfoldvalloss.png',logsAvg,['val_loss'])

#with pd.option_context('display.max_seq_items', None):
with pd.option_context('display.max_columns', None),pd.option_context('display.width',None),pd.option_context('display.max_colwidth',None):
    print(f"\nTraining Stats\n{logsAll[['binary_accuracy','loss','sensitivity','specificity','auc','precision','dice','kappa']].describe().transpose()}")
    print(f"\nValidation Stats\n{logsAll[['val_binary_accuracy','val_loss','val_sensitivity','val_specificity','val_auc','val_precision','val_dice','val_kappa']].describe().transpose()}")
    #print(f"\nTraining Stats\n{logsAll[['binary_accuracy','loss','sensitivity','specificity','auc','alt_auc','precision','dice','kappa']].describe().transpose()}")
    #print(f"\nValidation Stats\n{logsAll[['val_binary_accuracy','val_loss','val_sensitivity','val_specificity','val_auc','val_alt_auc','val_precision','val_dice','val_kappa']].describe().transpose()}")

