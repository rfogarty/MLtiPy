import numpy as np

# Since the ensemble was not set up as Keras layer, metrics were computed by hand.
# This simple routine compares the outcomes to the expected labels and computes
# the number correct and incorrect for each individual class.
def computeBinomialMetrics(outcomes,test_labels,paths,sampleStats,numModels) :
    correct=[0]*2
    incorrect=[0]*2
    minQuorum = int((numModels+1)/2) # If there are multiple models voting, there must be a min majority
    # Next compare predicted outcomes with the labels and add to appropriate
    # correct/incorrect table index.
    for pred,label,path in zip(outcomes,test_labels,paths):
        labelval = int(label)
        if labelval == 0 :
            numCorrect = numModels - pred
        else :
            numCorrect = pred
        if (numCorrect >= minQuorum) : correct[labelval] += 1
        else : incorrect[labelval] += 1
        sampleStats[path] += numCorrect
    
    return (correct,incorrect)


# This function is intended to print accuracies for multi-class problems
# given two arrays for correct and incorrect values (per class). Additionally,
# the total accuracy is also computed.
def printAccuracies(correct,incorrect) :
    totcorrect = 0
    totincorrect = 0
    for idx,(c,i) in enumerate(zip(correct,incorrect)) :
        totcorrect += c
        totincorrect += i
        if ((c + i) > 0) : acc = c / (c + i)
        else : acc = 'n/a'
        print(f"Digit {idx} accuracy : {acc} ({c},{i})")
    if ((totcorrect + totincorrect) > 0) : acc = totcorrect / (totcorrect + totincorrect)
    else : acc = 'n/a'
    print(f"Total accuracy : {acc}")
    return (acc,totcorrect,totincorrect)


# This function was developed to compute the upper and lower bounds
# or confidence limits as described in Data Mining Section 5.3
def computeBounds(acc,numgood,numbad,z=0.69) :
    N = numgood + numbad
    f = numbad/N
    z2 = z*z
    N2 = N*N
    f2 = f*f
    lower = (f + (z2/(2.0*N)) + (z * m.sqrt((f/N) - (f2/N) + (z2/(4.0*N2)))))/(1.0 + (z2/N))
    upper = (f + (z2/(2.0*N)) - (z * m.sqrt((f/N) - (f2/N) + (z2/(4.0*N2)))))/(1.0 + (z2/N))
    print(f"Total accuracy bounds : ({acc-lower}:{acc+upper})")
    return (upper,lower)


# Since the ensemble was not set up as Keras layer, metrics were computed by hand.
# This simple routine compares the outcomes to the expected labels and computes
# the number correct and incorrect for each individual class.
def computeMultinomialMetrics(outcomes,test_data,numClasses) :
    correct=[0]*numClasses
    incorrect=[0]*numClasses
    # First extract all of the labels:
    labels = np.concatenate([labelsubset for _, labelsubset in test_data], axis=0)
    # Next compare predicted outcomes with the labels and add to appropriate
    # correct/incorrect table index.
    for pred,label in zip(outcomes,labels):
        labelval = int(label)
        if pred == labelval : correct[labelval] += 1
        else : incorrect[labelval] += 1
    return (correct,incorrect)

