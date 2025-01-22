
import os,sys
from glob import glob as ls
import random
from paths import filter_filelist,list_images
from collections import defaultdict

##############################################################################
# Create random splits for data sets where it is not practical to create folds
#
# Created By: Ryan Botet Fogarty
# Last Edited: 2022.01.06 (added comments for clarity)
# 
##############################################################################

# Read a mapping file of the form:
#   MAST_22B_: 21
#   MAST_3A_: 80
#   MAST_3B_: 69
#   MAST_7A_: 166
# Each line represents a patient and how many samples
# are available for that patient
def parsePatchDesc(filepath) :
    datamap = {}
    numtype = 0
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines :
        name,number=line.split(':')
        number=int(number.strip())
        datamap[name] = number
        numtype += number
    return (datamap,numtype)


def parseBlocklist(filepath,trimpath=False) :
    blocklist=[]
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines :
        line = line.lstrip().rstrip()
        #if not line and not line.startswith('#') :
        if line and not line.startswith('#') :
            if trimpath: line = os.path.basename(line)
            blocklist.append(line)
    return blocklist


# If the classes are unbalanced:
#    datamap1 should be the minority class
#    datamap2 should be the majority class
def createSplit(datamap1,datamap2,num1,num2,holdoutProportion,evenSplit=True) :
    # Copy patients respresented in minority class (datamap1) and patients
    #               represented in majority class (datamap2) into lists that are
    #               then shuffled (we don't worry if a patient is in both min/maj
    #               classes and shuffled differently as that is handled below).
    datamap1_sh = list(datamap1)
    datamap2_sh = list(datamap2)
    random.shuffle(datamap1_sh)
    random.shuffle(datamap2_sh)

    holdoutSet=set()
    trainingSet=set()
    numOf1InHoldoutSet=0
    numOf2InHoldoutSet=0
    numOf1InTrainingSet=0
    numOf2InTrainingSet=0

    if evenSplit :
        # Strategy - iterate the list of files placing the first N in the test set
        evenSplitNum=min(num1,num2)*2 # *2 since evenSplitNum represents both classes
        holdoutSplitNum=holdoutProportion*evenSplitNum
        trainingSplitNum=(1.0-holdoutProportion)*evenSplitNum
        
        data1Pos=0
        data2Pos=0
        
        # First build the HoldoutSet
        while True:
           patient = None
           # Since the proportion gives us info on how uneven we are split, perhaps
           # use that as a very rough guide to select which side to sample from
           if numOf1InHoldoutSet > numOf2InHoldoutSet :
               # Since we have more minority class, sample from majority side
               patient=datamap2_sh[data2Pos]
               data2Pos += 1
           else :
               # Since we have fewer minority class, sample from that side
               patient=datamap1_sh[data1Pos]
               data1Pos += 1
           # If we've already processed this patient (which can happen since iterating min/maj
           # separately), simply skip over this case and continue.
           if patient in holdoutSet : continue
           else : # o.w. update holdoutSet
               # And now add patient to holdout set and accrue how many samples
               # are added to min/maj samples.
               if patient in datamap1 :
                  numOfPatientIn1 = datamap1[patient]
                  numOf1InHoldoutSet += numOfPatientIn1
               if patient in datamap2 :
                  numOfPatientIn2 = datamap2[patient]
                  numOf2InHoldoutSet += numOfPatientIn2
               # And add patient to Set!
               holdoutSet.add(patient)
           # If holdoutSet is sufficiently large, break and continue to building training set
           if (numOf1InHoldoutSet + numOf2InHoldoutSet) >= holdoutSplitNum : break
        
        # Next build the TrainingSet
        while True:
           patient = None
           # Since the proportion gives us info on how uneven we are split, perhaps
           # use that as a very rough guide to select which side to sample from
           if numOf1InTrainingSet > numOf2InTrainingSet :
               # Since we have more minority class, sample from majority side
               patient=datamap2_sh[data2Pos]
               data2Pos += 1
           else :
               # Since we have fewer minority class, sample from that side
               patient=datamap1_sh[data1Pos]
               data1Pos += 1
           if patient == None : break
           # Ensure that patient isn't in the holdout set or training set already
           # if so, skip this iteration
           if (patient in holdoutSet) or (patient in trainingSet) : continue
           else : # o.w. update holdoutSet
               if patient in datamap1 :
                  numOfPatientIn1 = datamap1[patient]
                  numOf1InTrainingSet += numOfPatientIn1
               if patient in datamap2 :
                  numOfPatientIn2 = datamap2[patient]
                  numOf2InTrainingSet += numOfPatientIn2
               # And add patient to Set!
               trainingSet.add(patient)
           if (numOf1InTrainingSet + numOf2InTrainingSet) >= trainingSplitNum : break
    else :
        stratifiedProportion = float(num1) / float(num2)
        # Ok doing a stratified split
        totalNum=num1+num2
        holdoutSplitNum=holdoutProportion*totalNum
        trainingSplitNum=(1.0-holdoutProportion)*totalNum

        # First build the HoldoutSet
        while True:
           patient = None
           # Since the proportion gives us info on how uneven we are split, perhaps
           # use that as a very rough guide to select which side to sample from
           if (numOf1InHoldoutSet*stratifiedProportion) > numOf2InHoldoutSet :
               # Since we have too many minority class, sample from majority side
               patient=datamap2_sh[data2Pos]
               data2Pos += 1
           else :
               # Since we have too few minority class, sample from that side
               patient=datamap1_sh[data1Pos]
               data1Pos += 1
           if patient in holdoutSet : continue
           else : # o.w. update holdoutSet
               if patient in datamap1 :
                  numOfPatientIn1 = datamap1[patient]
                  numOf1InHoldoutSet += numOfPatientIn1
               if patient in datamap2 :
                  numOfPatientIn2 = datamap2[patient]
                  numOf2InHoldoutSet += numOfPatientIn2
               # And add patient to Set!
               holdoutSet.add(patient)
           if (numOf1InHoldoutSet + numOf2InHoldoutSet) >= holdoutSplitNum : break
        
        # Next build the TrainingSet
        while True:
           patient = None
           # Since the proportion gives us info on how uneven we are split, perhaps
           # use that as a very rough guide to select which side to sample from
           if (numOf1InTrainingSet*stratifiedProportion) > numOf2InTrainingSet :
               # Since we have too many minority class, sample from majority side
               patient=datamap2_sh[data2Pos]
               data2Pos += 1
           else :
               # Since we have too few minority class, sample from that side
               patient=datamap1_sh[data1Pos]
               data1Pos += 1
           if patient == None : break
           if (patient in holdoutSet) or (patient in trainingSet) : continue
           else : # o.w. update holdoutSet
               if patient in datamap1 :
                  numOfPatientIn1 = datamap1[patient]
                  numOf1InTrainingSet += numOfPatientIn1
               if patient in datamap2 :
                  numOfPatientIn2 = datamap2[patient]
                  numOf2InTrainingSet += numOfPatientIn2
               # And add patient to Set!
               trainingSet.add(patient)
           if (numOf1InTrainingSet + numOf2InTrainingSet) >= trainingSplitNum : break
    # Convert sets to lists that are first sort than shuffled in deterministic order (for repeatability)
    holdoutSet = list(holdoutSet)
    holdoutSet.sort()
    random.shuffle(holdoutSet)
    trainingSet = list(trainingSet)
    trainingSet.sort()
    random.shuffle(trainingSet)
    return (holdoutSet,trainingSet,numOf1InHoldoutSet,numOf2InHoldoutSet,numOf1InTrainingSet,numOf2InTrainingSet)


def fixStratification(set1n,set2n,beta,evenSplit=True) :
    smalln=min(set1n,set2n)
    largen=max(set1n,set2n)
    if evenSplit :
        return (smalln,smalln)
    else :
        if smalln/largen > beta :
            diffn=smalln - beta*largen
            return (smalln-diffn,largen)
        else :
            diffn=largen-smalln/beta
            return (smalln,largen-diffn)


def fixHoldout(holdout,training,alpha) :
    if holdout/(training+holdout) > alpha :
        diffn=holdout - training*alpha/(1.0-alpha)
        return ('holdout',holdout-diffn,training)
    else :
        diffn=training + holdout - holdout/alpha
        return ('training',holdout,training-diffn)


def smartReshuffle(elements,usageMap,numToKeep) :
    random.shuffle(elements)
    # Algo:
    #   Shuffle elements
    #   Ensure all elements are initialized in usageMap (this is done implicitly by using a defaultdict!)
    #   Rotate elements with fewest usages to top of shuffled list, most used move to bottom
    #   Lastly, update the usageMap for each element which we will use in this fold

    # Note that the "timsort" algorithm is stable, and thus order of shuffled elements should not change
    # for those that have the same usage number
    elements.sort(key=lambda e: usageMap[e])
    for i in range(numToKeep) :
        element = elements[i]
        usageMap[element] += 1
    

# Create a list of files from a minority and majority directory
# given a "split" as generated by the createSplit() method.
def createFileLists(minDir,majDir,split,usageMap,holdoutProportion,evenSplit=True,blocklistpath='blocklist.txt') :
    holdoutSet,trainingSet,numOf1InHoldoutSet,numOf2InHoldoutSet,numOf1InTrainingSet,numOf2InTrainingSet = split
    if os.path.exists(blocklistpath) :
        blocklist = parseBlocklist(blocklistpath)
    else :
        blocklist = None
 
    print(f'HoldoutSet: {holdoutSet}')
    print(f'TrainingSet: {trainingSet}')
    print('Sizes before stratification and balancing')
    print(f'numOf1InHoldoutSet={numOf1InHoldoutSet}')
    print(f'numOf2InHoldoutSet={numOf2InHoldoutSet}')
    print(f'numOf1InTrainingSet={numOf1InTrainingSet}')
    print(f'numOf2InTrainingSet={numOf2InTrainingSet}')
    # Start by adjusting numbers of examples we would like to keep to fix stratification and holdout proportions exactly
    beta=(numOf1InHoldoutSet+numOf1InTrainingSet)/(numOf2InHoldoutSet+numOf2InTrainingSet)
    numOf1InHoldoutSet,numOf2InHoldoutSet=fixStratification(numOf1InHoldoutSet,numOf2InHoldoutSet,beta,evenSplit)
    numOf1InTrainingSet,numOf2InTrainingSet=fixStratification(numOf1InTrainingSet,numOf2InTrainingSet,beta,evenSplit)
    setAdjusted,numOf1InHoldoutSet,numOf1InTrainingSet=fixHoldout(numOf1InHoldoutSet,numOf1InTrainingSet,holdoutProportion)
    # And readjust stratification for reduced (from holdout prop) set
    if setAdjusted == 'holdout' :
        numOf1InHoldoutSet,numOf2InHoldoutSet=fixStratification(numOf1InHoldoutSet,numOf2InHoldoutSet,beta,evenSplit)
    else :
        numOf1InTrainingSet,numOf2InTrainingSet=fixStratification(numOf1InTrainingSet,numOf2InTrainingSet,beta,evenSplit)
    numOf1InHoldoutSet=int(numOf1InHoldoutSet)
    numOf2InHoldoutSet=int(numOf2InHoldoutSet)
    numOf1InTrainingSet=int(numOf1InTrainingSet)
    numOf2InTrainingSet=int(numOf2InTrainingSet)
    print('Sizes after stratification and balancing')
    print(f'numOf1InHoldoutSet={numOf1InHoldoutSet}')
    print(f'numOf2InHoldoutSet={numOf2InHoldoutSet}')
    print(f'numOf1InTrainingSet={numOf1InTrainingSet}')
    print(f'numOf2InTrainingSet={numOf2InTrainingSet}')
    holdoutFiles1=[]
    holdoutFiles2=[]
    for holdoutprefix in holdoutSet :
        minFiles=ls(minDir + '/' + holdoutprefix + '*')
        majFiles=ls(majDir + '/' + holdoutprefix + '*')
        for minFile in minFiles :
            holdoutFiles1.append(minFile)
        for majFile in majFiles :
            holdoutFiles2.append(majFile)
    if blocklist is not None :
        holdoutFiles1=filter_filelist(holdoutFiles1,blocklist=blocklist)
        holdoutFiles2=filter_filelist(holdoutFiles2,blocklist=blocklist)
    smartReshuffle(holdoutFiles1,usageMap,numOf1InHoldoutSet)
    smartReshuffle(holdoutFiles2,usageMap,numOf2InHoldoutSet)
    #random.shuffle(holdoutFiles1)
    #random.shuffle(holdoutFiles2)
    del holdoutFiles1[numOf1InHoldoutSet:]
    del holdoutFiles2[numOf2InHoldoutSet:]

    trainingFiles1=[]
    trainingFiles2=[]
    for trainingprefix in trainingSet :
        minFiles=ls(minDir + '/' + trainingprefix + '*')
        majFiles=ls(majDir + '/' + trainingprefix + '*')
        for minFile in minFiles :
            trainingFiles1.append(minFile)
        for majFile in majFiles :
            trainingFiles2.append(majFile)
    if blocklist is not None :
        trainingFiles1=filter_filelist(trainingFiles1,blocklist=blocklist)
        trainingFiles2=filter_filelist(trainingFiles2,blocklist=blocklist)
    smartReshuffle(trainingFiles1,usageMap,numOf1InTrainingSet)
    smartReshuffle(trainingFiles2,usageMap,numOf2InTrainingSet)
    #random.shuffle(trainingFiles1)
    #random.shuffle(trainingFiles2)
    del trainingFiles1[numOf1InTrainingSet:]
    del trainingFiles2[numOf2InTrainingSet:]
    # Consider returning this as 4 lists instead
    return ((holdoutFiles1,holdoutFiles2),(trainingFiles1,trainingFiles2))

#def makePatchDescription(dataDir,splitPattern,blocklistname='blocklist.txt') :
def makePatchDescription(dataDir,splitPattern,blocklistpath='blocklist.txt') :
    images = list(list_images(dataDir))
    images.sort()
    #blocklistpath = os.path.join(dataDir,blocklistname)
    if os.path.exists(blocklistpath) :
        blocklist = parseBlocklist(blocklistpath)
        images = filter_filelist(images,blocklist=blocklist)
        images.sort()

    # Now construct map by splitting at splitPattern
    # Using special collection class defaultdict so initialization
    #    of new dictionary elements is automatic.
    patchmap=defaultdict(int)
    for im in images :
        prefix=im.split(splitPattern,1)[0]
        prefix=os.path.basename(prefix)
        patchmap[prefix] += 1

    #imageset=set()
    #patchmap={}
    #for im in images :
    #    prefix=im.split(splitPattern,1)[0]
    #    if prefix in imageset :
    #        patchmap[prefix] += 1
    #    else :
    #        imageset.add(prefix)
    #        patchmap[prefix] = 0
    return (patchmap,len(images))


def makeSplits(prefix,minDir,majDir,holdoutProportion=0.2,numSets=10,patchDescMin=None,patchDescMaj=None,splitPattern='stack') :
    #minNum=2484
    #majNum=5143
    patchDescMinPath='' if patchDescMin is None else os.path.join(minDir,patchDescMin)
    patchDescMajPath='' if patchDescMaj is None else os.path.join(majDir,patchDescMaj)

    if os.path.exists(patchDescMinPath) :
        minDatamap,minNum=parsePatchDesc(patchDescMinPath)
    else :
        print('Patch Description for minority class not found, attempting to create...')
        minDatamap,minNum=makePatchDescription(minDir,splitPattern,blocklistpath='blocklist.txt')
        print(minDatamap)
    
    if os.path.exists(patchDescMajPath) :
        majDatamap,majNum=parsePatchDesc(patchDescMajPath)
    else :
        print('Patch Description for majority class not found, attempting to create...')
        majDatamap,majNum=makePatchDescription(majDir,splitPattern,blocklistpath='blocklist.txt')
        print(majDatamap)

    
    # First create numSets random splits
    splits=[]
    for i in range(numSets) :
        splits.append((createSplit(minDatamap,majDatamap,minNum,majNum,holdoutProportion)))
    
    #print('Splits:')
    #print(splits)
    
    # Now clean up the stratification and proportions to be as desired and
    # create file lists given the splits
    usageMap=defaultdict(int)
    for idx,split in enumerate(splits) :
        holdoutFiles,trainingFiles=createFileLists(minDir,majDir,split,usageMap,holdoutProportion,evenSplit=True,blocklistpath='blocklist.txt')
        holdoutfile=prefix + str(idx) + 'Holdout' + '.list'
        trainingfile=prefix + str(idx) + 'Training' + '.list'
        with open(holdoutfile, 'w') as f:
            for h in holdoutFiles[0] :
                print(h, file=f)
            for h in holdoutFiles[1] :
                print(h, file=f)
        with open(trainingfile, 'w') as f:
            for t in trainingFiles[0] :
                print(t, file=f)
            for t in trainingFiles[1] :
                print(t, file=f)
    
    print('Splits done')


def makeSplitsProstatePathology(prefix,holdoutProportion=0.2,numSets=10) :
    minDir='GS4'
    majDir='GS3'
    #'PatchDescriptionGS4.txt'
    #'GS3/PatchDescriptionGS3.txt'
    makeSplits(prefix,minDir,majDir,holdoutProportion=holdoutProportion,numSets=numSets)

# Algorithm for creating splits in this Python file is this:
#
# 1. Read in mapping files of patients with number of labeled glands for each Gleason Class
# 2. Create N splits by doing the following:
#    2.1 Randomize or Shuffle the order of these patients for each class (GS3 and GS4)
#    2.2 Set a holdout proportion (default = 0.2 or 20%)
#    2.3 Determine the stratification in the original set (or force to 50% with boolean "evenSplit")
#    2.4 Iterate the list of shuffled patients
#        2.4.1 If we too few class 1 data (vs. class 2 data in holdout set) read the next patient from class 1 list
#              o.w. read patient from class 2 list
#        2.4.2.a If patient exists in class 1, add that many labeled glands to holdout set
#        2.4.2.b If patient exists in class 2, add that many labeled glands to holdout set
#        2.4.3 Break from loop when we have enough data in the holdout set as set by holdout proportion
#    2.5 Iterate the rest of the lists of shuffled patients to create the training sets
#        2.5.1 Add data from both class 1 and class 2 lists to training sets
#        2.5.2 Break early if we have enough data to satisfy (1-holdoutProportion) ratio - i.e. all patients may not end up in a split
# 3. For each split
#    3.1 Tune the numbers of the holdout set and training sets to guarantee the same stratification
#    3.2 Tune the proportion balance of holdout to training to ensure matches our holdoutProportion criteria
#    3.3 Generate list of images for holdout set
#        3.3.1 Read in patients in holdout set and perform globbing read on class directories
#        3.3.2 Shuffle the file list for each class
#        3.3.3 Reduce the number of files to the tuned sizes (after stratification and proportion balancing)
#    3.4 Generate list of images for training set
#        3.4.1 Read in patients in training set and perform globbing read on class directories
#        3.4.2 Shuffle the file list for each class
#    3.5 Write out holdout file list to file
#    3.6 Write out training file list to file
if __name__ == '__main__':
    random.seed(23)
    #makeSplits('split')
    makeSplitsProstatePathology('split')

