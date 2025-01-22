
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import os

def openCSV(filename) :
    return pd.read_csv(filename) 


def loadLogs() :
    logs = []
    logs.append(openCSV('log0.csv'))
    logs.append(openCSV('log1.csv'))
    logs.append(openCSV('log2.csv'))
    logs.append(openCSV('log3.csv'))
    logs.append(openCSV('log4.csv'))
    logs.append(openCSV('log5.csv'))
    logs.append(openCSV('log6.csv'))
    logs.append(openCSV('log7.csv'))
    logs.append(openCSV('log8.csv'))
    logs.append(openCSV('log9.csv'))
    if os.path.exists('log19.csv') :
        logs.append(openCSV('log10.csv'))
        logs.append(openCSV('log11.csv'))
        logs.append(openCSV('log12.csv'))
        logs.append(openCSV('log13.csv'))
        logs.append(openCSV('log14.csv'))
        logs.append(openCSV('log15.csv'))
        logs.append(openCSV('log16.csv'))
        logs.append(openCSV('log17.csv'))
        logs.append(openCSV('log18.csv'))
        logs.append(openCSV('log19.csv'))
        logs3D=np.array([logs[0],logs[1],logs[2],logs[3],logs[4],logs[5],logs[6],logs[7],logs[8],logs[9],logs[10],logs[11],logs[12],logs[13],logs[14],logs[15],logs[16],logs[17],logs[18],logs[19]])
    else :
        logs3D=np.array([logs[0],logs[1],logs[2],logs[3],logs[4],logs[5],logs[6],logs[7],logs[8],logs[9]])
    
    logs2D=np.mean(logs3D,axis=0)
    logsAll = pd.concat(logs)
    logsAvg=pd.DataFrame(logs2D,columns=logs[0].columns.values)
    logsParallel = pd.concat(logs,axis=1)
    return (logsAll,logsParallel,logsAvg,logs)


def boxplotStat(filename,dataframe,stats,relabel=True) :
    pl.clf()
    dataframe.boxplot(stats)
    pl.title(stats)
    if relabel :
        indices=[r+1 for r in range(np.shape(dataframe.loc[:,stats])[1])]
        indiceslab=[r for r in range(np.shape(dataframe.loc[:,stats])[1])]
        labels=list(np.array(indiceslab).astype(str))
        indices.insert(0,0)
        labels.insert(0,'')
        pl.xticks(indices,labels)
    pl.savefig(filename)


def lineplotStat(filename,dataframe,stats,relabel=True) :
    pl.clf()
    pl.plot(dataframe.loc[:,stats])
    pl.title(stats)
    #if relabel :
    #    indices=[r+1 for r in range(np.shape(dataframe.loc[:,stats])[1])]
    #    indiceslab=[r for r in range(np.shape(dataframe.loc[:,stats])[1])]
    #    labels=list(np.array(indiceslab).astype(str))
    #    indices.insert(0,0)
    #    labels.insert(0,'')
    #    pl.xticks(indices,labels)
    pl.savefig(filename)


def boxplotAccuracies(filename,dataframe) :
    pl.clf()
    dataframe.boxplot(['binary_accuracy','val_binary_accuracy'])
    pl.title('Accuracies')
    pl.savefig(filename)


def printAccuracyStats(dataframe) :
    dataframe.binary_accuracy.describe()


def printValAccuracyStats(dataframe) :
    dataframe.val_binary_accuracy.describe()


def printStats(dataframe,stats) :
    dataframe.loc[:,stats].describe()


#Sensitivity TP/(TP+FN)	Specificity TN/(TN+FP)	Precision - TP/(TP+FP)	F1 (Dice) 2*P*S/(P+S)
def derivedTrainStats(dataframe) :
   TP=dataframe.true_positives
   TN=dataframe.true_negatives
   FP=dataframe.false_positives
   FN=dataframe.false_negatives
   sensitivity=TP/(TP+FN)
   specificity=TN/(TN+FP)
   auc=(specificity+sensitivity)/2
   #print(f'sensitivity.shape ({sensitivity.dtype}):{sensitivity.shape}')
   #print(f'specificity.shape ({specificity.dtype}):{specificity.shape}')
   #print(f'auc.shape ({auc.dtype}):{auc.shape}')
   precision=TP/(TP+FP)
   dice=precision*sensitivity/(precision+sensitivity)
   dice=dice.mul(2.0)
   kappa=((TP*TN)-(FN*FP))/(((TP+FP)*(FP+TN))+((TP+FN)*(FN+TN)))
   kappa=kappa.mul(2.0)
   altStats=pd.DataFrame({'sensitivity':sensitivity,'specificity':specificity,'alt_auc':auc,'precision':precision,'dice':dice,'kappa':kappa})
   return altStats


def derivedValStats(dataframe) :
   TP=dataframe.val_true_positives
   TN=dataframe.val_true_negatives
   FP=dataframe.val_false_positives
   FN=dataframe.val_false_negatives
   sensitivity=TP/(TP+FN)
   specificity=TN/(TN+FP)
   auc=(specificity+sensitivity)/2
   #print(f'sensitivity.shape ({sensitivity.dtype}):{sensitivity.shape}')
   #print(f'specificity.shape ({specificity.dtype}):{specificity.shape}')
   #print(f'auc.shape ({auc.dtype}):{auc.shape}')
   precision=TP/(TP+FP)
   dice=precision*sensitivity/(precision+sensitivity)
   dice=dice.mul(2.0)
   kappa=((TP*TN)-(FN*FP))/(((TP+FP)*(FP+TN))+((TP+FN)*(FN+TN)))
   kappa=kappa.mul(2.0)
   altStats=pd.DataFrame({'val_sensitivity':sensitivity,'val_specificity':specificity,'val_alt_auc':auc,'val_precision':precision,'val_dice':dice,'val_kappa':kappa})
   return altStats


if __name__ == "__main__" :
    logsAll,logs=loadLogs()

