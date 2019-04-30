import numpy as np

import os
import math
import copy
import shutil
import scipy
import random
from enum import Enum
from tqdm import tqdm

# Visualization libraries
USE_SEABORN = False
import matplotlib.pyplot as plt
if USE_SEABORN:
    import seaborn as sns
    sns.set()  # Set the seaborn defaults

# Pruning libraries
import keras
import kerassurgeon as ks

# Clustering libraries
# from kMeansDTW import KMeansDTW, dtw_distances
from dtaidistance import dtw  # Also uses euclidean distance by default

from sklearn.cluster import KMeans, MeanShift
import sklearn.mixture

from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist, squareform

# Suppress warning from DTAI
import logging
logger = logging.getLogger("be.kuleuven.dtai.distance")
logger.disabled = True

# Import configuration
from config import *


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~ Convenience classes ~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class Modes(Enum):
    # Importance selection modes
    LOSS = 0
    IMPORTANCE = 1

    # Modes - prune()
    PRUNE = 0
    ADJUST = 1

    # Modes - get_filter_list_from_file()
    COMPUTE_RANDOM = 0
    COMPUTE_PERCENTILE = 1
    COMPUTE_REPRESENTATIVE = 2

    # Importance modes - get_filter_list()
    PERCENTILE_MAXIMUM = 0
    PERCENTILE_MINIMUM = 1
    SORTED_IMPORTANCE_MAXIMUM = 2
    SORTED_IMPORTANCE_MINIMUM = 3
    CLUSTER_REPRESENTATIVES = 4

    # Sub-modes - get_filter_list_from_file()
    MINIMUM = 0
    MAXIMUM = 1
    MEAN = 2

    # Data modes
    SELECTED_EXAMPLES = 0
    ALL_EXAMPLES = 1

    # Fast modes
    FULL = 0
    PARTIAL = 1
    MINIMAL = 2


class Point:
    def __init__(self, initx, inity):
        self.x = initx
        self.y = inity

    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff*self.x - x_diff*self.y + p2.x*p1.y - p2.y*p1.x)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~ Functions for Data Loading ~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def create_dataset(dataset, look_back, steps):
    dataX, dataY = [], []
    l = int((len(dataset) - look_back) / steps)
    derivative = steps
    cols = dataset.shape[1]
    for i in range(l):
        if i == 0:
            print('first iteration')
            prei = i
            temp = []
            temp_train = np.asarray(dataset.iloc[prei:(prei + look_back), 0])
            temp.append(np.reshape(temp_train, (temp_train.shape[0], 1)))
            for i in range(0, derivative):
                temp_train = np.ediff1d(temp_train)
                temp_train = np.append(temp_train, np.mean(np.fabs(temp_train)))
                temp_train = np.reshape(temp_train, (temp_train.shape[0], 1))
                temp.append(temp_train)
            for index in range(1, cols):
                temp_col = dataset.iloc[prei:(prei + look_back), index]
                temp_col = np.reshape(temp_col, (temp_col.shape[0], 1))
                temp.append(temp_col)
            temp = np.asarray(temp)
            temp = totuple(temp)
            temp = np.column_stack(temp)
            dataX.append(temp)
            if steps == 1:
                temp2 = dataset.iloc[(prei + look_back):(prei + look_back + steps), 0].values
                dataY.append(np.reshape(temp2, (temp2.shape[0], 1)))
            else:
                dataY.append(dataset.iloc[(prei + look_back):(prei + look_back + steps), 0])

            continue

        prei = prei + steps
        f = dataset.iloc[(prei + look_back):(prei + look_back + steps), 0]
        if len(f) < steps:
            continue
        temp = []
        temp_train = np.asarray(dataset.iloc[prei:(prei + look_back), 0])
        temp.append(np.reshape(temp_train, (temp_train.shape[0], 1)))
        for i in range(0, derivative):
            temp_train = np.ediff1d(temp_train)
            temp_train = np.append(temp_train, np.mean(np.fabs(temp_train)))
            temp_train = np.reshape(temp_train, (temp_train.shape[0], 1))
            temp.append(temp_train)
        for index in range(1, cols):
            temp_col = dataset.iloc[prei:(prei + look_back), index]
            temp_col = np.reshape(temp_col, (temp_col.shape[0], 1))
            temp.append(temp_col)
        temp = np.asarray(temp)
        temp = totuple(temp)
        temp = np.column_stack(temp)
        dataX.append(temp)
        if steps == 1:
            temp2 = dataset.iloc[(prei + look_back):(prei + look_back + steps), 0].values
            dataY.append(np.reshape(temp2, (temp2.shape[0], 1)))
        else:
            dataY.append(dataset.iloc[(prei + look_back):(prei + look_back + steps), 0])

    return np.array(dataX, dtype=np.float32), np.array(dataY, dtype=np.float32)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~ Inverse Optimization and Adversarial examples ~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def smoothSampling(numVals, numChannels, mean=0.0, std=0.05):
    finalSeries = []
    for channel in range(numChannels):
        stdDev = 0.05 + np.random.normal(scale=std)  # [0, 1)
        result = []
        y = np.random.normal(loc=mean, scale=stdDev)
        for _ in range(numVals):
            result.append(y)
            y += np.random.normal(loc=mean, scale=stdDev)
        result = np.array(result)

        # Normalize
        result = scipy.stats.zscore(result)  # @UndefinedVariable

        finalSeries.append(result)

    return np.stack(finalSeries, axis=1)


def smooth(y, kernelWidth=3):
    kernel = np.ones(kernelWidth)/kernelWidth
    ySmooth = np.convolve(y, kernel, mode='same')
    return ySmooth


def smoothConsistentSampling(numVals, mean=0.0, stdDev=0.05):
    y = np.random.normal(loc=mean, scale=stdDev)
    result = []
    for _ in range(numVals):
        result.append(y)
        y += np.random.normal(loc=mean, scale=stdDev)
    return np.array(result)


def computeNetworkLoss(kerasModel, X, y, verbose=True):
    loss = kerasModel.evaluate(X, np.squeeze(y), verbose=0)  # Silent
    if verbose:
        print("Loss: %s | Loss metric: %s" % (str(loss), kerasModel.metrics_names))
    return loss


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~ Filter Pruning ~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def indiceStringToList(string):
    """
    for example:
    turns 1,2,63,7;5,2,986;305,3;
    into [[1,2,63,7], [5,2,986], [305,3], []]
    """
    output = [[int(char) for char in row.split(',') if char != ''] for row in string.split(';')]
    return output


def listToIndiceString(my_list):
    """
    for example:
    turns [[1,2,63,7], [5,2,986], [305,3], []]
    into 1,2,63,7;5,2,986;305,3;
    """
    output = ";".join([",".join([str(int(item)) for item in row]) for row in my_list])
    return output


def pruneNetwork(session, kerasModel, filterToRemove):
    currentLayer = 0
    with session.graph.as_default():
        surgeon = ks.Surgeon(kerasModel, copy=True)
        for layerIdx, layer in enumerate(kerasModel.layers):
            if not ("flatten" in layer.name or "pool" in layer.name or "input" in layer.name or "dense" in layer.name):
                # Remove the filters described in the list
                if (currentLayer < len(filterToRemove)) and (len(filterToRemove[currentLayer]) > 0):
                    print("Removing filters from layer: %s" % layer.name)
                    surgeon.add_job('delete_channels', kerasModel.layers[layerIdx], channels=filterToRemove[currentLayer])
                currentLayer += 1

        new_model = surgeon.operate()
        new_model.compile(loss=kerasModel.loss, metrics=kerasModel.metrics, optimizer=kerasModel.optimizer)

    return new_model


def adjustWeights(session, kerasModel, filterToAdjust):
    currentLayer = 0
    with session.graph.as_default():
        new_model = keras.models.clone_model(kerasModel)
        new_model.set_weights(kerasModel.get_weights())
        for layer in new_model.layers:
            if not ("flatten" in layer.name or "pool" in layer.name or "input" in layer.name or "dense" in layer.name):
                # Obtain layer weights
                weights = layer.get_weights()
                if (currentLayer < len(filterToAdjust)) and (len(filterToAdjust[currentLayer]) > 0):
                    print("Updated filter for layer: %s" % layer.name)

                    # Weights shape in TF: [kernel_size, input_channels, output_channels] (For time-series)
                    # [kernel_height, kernel_width, input_channels, output_channels] (For images)
                    for filterIdx in filterToAdjust[currentLayer]:
                        print("Setting filter %d from layer %s to zero" % (filterIdx, layer.name))

                        # Set the weights to zero
                        for featureIdx in range((len(weights[0][0]))):
                            for kernelIdx in range(len(weights[0])):
                                weights[0][kernelIdx][featureIdx][filterIdx] = 0

                        # Set the biases to zero
                        weights[1][filterIdx] = 0

                    # Set the new weights
                    layer.set_weights(weights)
                currentLayer += 1
        new_model.compile(loss=kerasModel.loss, metrics=kerasModel.metrics, optimizer=kerasModel.optimizer)

    return new_model


def finetunePrunedModel(session, model, visualizationToolbox, epochNumber):
    with session.graph.as_default():
        if epochNumber > 0:
            print("Training Model!")
            model.fit(x=visualizationToolbox.trainX, y=np.reshape(visualizationToolbox.trainY, visualizationToolbox.trainY.shape[0]), epochs=epochNumber, verbose=0, validation_split=0.2)

        print("Evaluating Models")
        print("Metrics:", visualizationToolbox.model.metrics_names)

        score = visualizationToolbox.model.evaluate(x=visualizationToolbox.testX, y=np.reshape(visualizationToolbox.testY, visualizationToolbox.testY.shape[0]), verbose=1)
        print("Default model:", score)

        scoreP = model.evaluate(x=visualizationToolbox.testX, y=np.reshape(visualizationToolbox.testY, visualizationToolbox.testY.shape[0]), verbose=1)
        print("Pruned Model:", scoreP)

    results = np.subtract(scoreP, score)
    print("Results:", results)

    return score, scoreP


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~ Filter Selection ~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def computeIndices(importanceValues, importanceMode, numberOfFilter, layerSelection, numberOfConvs):
    result = np.empty(numberOfConvs, dtype=object)
    for i in range(len(result)):
        result[i] = []

    for layerIdx in range(len(result)):
        if not layerIdx in layerSelection:
            continue

        if importanceMode == Modes.PERCENTILE_MINIMUM.value or importanceMode == Modes.PERCENTILE_MAXIMUM.value:
            # Percentile
            perc = np.percentile(importanceValues[layerIdx], numberOfFilter)
            for filterIdx, filterIm in enumerate(importanceValues[layerIdx]):
                if importanceMode == Modes.PERCENTILE_MAXIMUM.value and filterIm > perc:
                    result[layerIdx].append(filterIdx)
                if importanceMode == Modes.PERCENTILE_MINIMUM.value and filterIm < perc:
                    result[layerIdx].append(filterIdx)

        if importanceMode == Modes.SORTED_IMPORTANCE_MINIMUM.value or importanceMode == Modes.SORTED_IMPORTANCE_MAXIMUM.value:
            # Sorted by importance with fixed value
            sortedIdxs = np.argsort(importanceValues[layerIdx])
            if importanceMode == Modes.SORTED_IMPORTANCE_MAXIMUM.value:
                result[layerIdx] = sortedIdxs[-numberOfFilter:]
            if importanceMode == Modes.SORTED_IMPORTANCE_MINIMUM.value:
                result[layerIdx] = sortedIdxs[:numberOfFilter]

    return result


def computeRepresentatives(visualizationToolbox, serviceOutput, layerSelection, convIds):
    result = np.empty(len(convIds), dtype=object)
    for i in range(len(result)):
        result[i] = []

    for layerIdx in range(len(result)):
        if not layerIdx in layerSelection:
            continue

        maxCluster = 0
        filterList = serviceOutput[convIds[layerIdx]]
        for filterO in filterList:
            if filterO[visualizationToolbox.filterClusterIdx] > maxCluster:
                maxCluster = filterO[visualizationToolbox.filterClusterIdx]

        result[layerIdx] = np.zeros(maxCluster+1)
        currentClusterImportance = np.zeros(maxCluster+1)

        for filterIdx, filterO in enumerate(filterList):
            if filterO[visualizationToolbox.filterImportanceIdx] > currentClusterImportance[filterO[6]]:
                maxCluster = filterIdx
                currentClusterImportance[filterO[visualizationToolbox.filterClusterIdx]] = filterO[visualizationToolbox.filterImportanceIdx]
                result[layerIdx][filterO[visualizationToolbox.filterClusterIdx]] = filterIdx
    return result


def matchRepresentatives(representativesExamples):
    resultMatched = np.empty(len(representativesExamples[0]))
    for i in range(len(resultMatched)):
        resultMatched[i] = [[]]

    for example in representativesExamples:
        for layerIdx, layerO in enumerate(example):
            for repIdx, rep in enumerate(layerO):
                if repIdx > len(resultMatched[layerIdx]):
                    resultMatched[layerIdx].append([])
                else:
                    resultMatched[layerIdx][repIdx] = rep

    result = np.empty(len(representativesExamples[0]))

    for i in range(len(result)):
        result[i] = []

    for layerIdx, layerO in enumerate(resultMatched):
        for clusterO in layerO:
            ids, occ = np.unique(clusterO, return_counts=True)
            rep = ids[np.argmax(occ)]
            result[layerIdx].append(rep)

    return result


def randomUseless(visualizationToolbox, percentile):
    '''
    Get random filters
    '''
    layerList = []
    layers = visualizationToolbox.model.layers
    layers = [layer for layer in layers if not (("flatten" in layer.name) or ("pool" in layer.name) or ("input" in layer.name) or ("dense" in layer.name))]
    for l in range(len(layers)):
        numFilters = layers[l].output_shape[2]
        prunedNumber = int(numFilters*percentile/100)
        layerList.append(random.sample(range(0, numFilters), prunedNumber))

    return layerList


def computeRemovementSets(visualizationToolbox, path, p):
    '''
    Get important filters
    '''
    importanceStats = np.load(path)
    allSortedMin = sortFiltersArg(visualizationToolbox, importanceStats, Modes.MINIMUM.value)
    allSortedMax = sortFiltersArg(visualizationToolbox, importanceStats, Modes.MAXIMUM.value)
    allSortedMean = sortFiltersArg(visualizationToolbox, importanceStats, Modes.MEAN.value)

    removedPercentileMin = removePercentile(allSortedMin, p)
    removedPercentileMax = removePercentile(allSortedMax, p)
    removedPercentileMean = removePercentile(allSortedMean, p)

    result = [removedPercentileMin, removedPercentileMax, removedPercentileMean]
    return result


def sortFiltersArg(visualizationToolbox, importanceStats, mode):
    # mode 0 for min, 1 for max and 2 for mean
    index = []
    importance = []
    for l in range(importanceStats.shape[0]):
        layerName = visualizationToolbox.layerNames[l]
        if not (("flatten" in layerName) or ("dense" in layerName) or ("input" in layerName) or ("pool" in layerName)):
            listModded = [x[mode] for x in importanceStats[l]]
            index.append(np.argsort(listModded))
            importance.append(np.sort(listModded))

    result = [index, importance]
    return result


def removePercentile(sortedListImportance, percentile):
    # mode 0 for min, 1 for max, and 2 for mean
    index = []
    importance = []
    for l in range(len(sortedListImportance[0])):
        thresh = np.percentile(sortedListImportance[1][l], percentile)
        filterList = []
        imp = []
        for f in range(len(sortedListImportance[0][l])):
            if sortedListImportance[1][l][f] < thresh:
                filterList.append(sortedListImportance[0][l][f])
                imp.append(sortedListImportance[1][l][f])
        index.append(filterList)
        importance.append(imp)

    result = [index, importance]
    return result


def computeMostTimesRepresentives(visualizationToolbox, path, dataset):
    '''
    Get representative filters
    '''
    serviceOutput = visualizationToolbox.loadData(dataset=dataset, fastMode=Modes.PARTIAL.value)
    representivesStats = np.load(path)
    occuranceList = np.zeros(len(serviceOutput[visualizationToolbox.layerNamesIdx]), dtype=object)
    finalRepresentives = np.zeros(len(serviceOutput[visualizationToolbox.layerNamesIdx]), dtype=object)

    for layerId in range(visualizationToolbox.dataStartingIdx, len(serviceOutput)-1):
        numFilters = int(len(serviceOutput[layerId]))
        filterList = np.zeros(numFilters)
        occuranceList[layerId-visualizationToolbox.dataStartingIdx] = filterList
        maxRepresentants = int(np.max([x[visualizationToolbox.filterClusterIdx] for x in serviceOutput[layerId]]))
        finalRepresentives[layerId-visualizationToolbox.dataStartingIdx] = maxRepresentants

    for exampleIdx in range(len(representivesStats)):
        for layerIdx in range(len(representivesStats[exampleIdx])):
            for filterIdx in range(len(representivesStats[exampleIdx][layerIdx])):
                filterNumber = int(representivesStats[exampleIdx][layerIdx][filterIdx])
                occuranceList[layerIdx][filterNumber] += 1

    for layerIdx in range(len(occuranceList)):
        numTopRepresentants = finalRepresentives[layerIdx]
        sortedRepresentants = np.argsort(occuranceList[layerIdx])
        finalRepresentives[layerIdx] = sortedRepresentants[-numTopRepresentants:]

    return finalRepresentives


def computeRepresentives(visualizationToolbox, serviceOutput, valueIdx):
    representativeList = np.zeros(len(serviceOutput[visualizationToolbox.layerNamesIdx])-1, dtype=object)
    for layerId in range(visualizationToolbox.dataStartingIdx, len(serviceOutput)-1):
        numClusters = int(np.max([x[visualizationToolbox.filterClusterIdx] for x in serviceOutput[layerId]])) + 1
        representativeFilters = np.zeros(numClusters)
        representativeFiltersImp = np.full(numClusters, fill_value=-1)

        for filterIdx, filterOb in enumerate(serviceOutput[layerId]):
            filterImp = filterOb[valueIdx]
            filterCluster = filterOb[visualizationToolbox.filterClusterIdx]
            if filterImp > representativeFiltersImp[filterCluster]:
                representativeFilters[filterCluster] = filterIdx
                representativeFiltersImp[filterCluster] = filterImp

        representativeList[layerId-visualizationToolbox.dataStartingIdx] = representativeFilters
    return representativeList


def keepOnlyRepresentants(visualizationToolbox, representants, dataset):
    serviceOutput = visualizationToolbox.loadData(dataset=dataset, fastMode=Modes.PARTIAL.value)
    result = []
    for l in range(len(representants)):
        layerName = serviceOutput[visualizationToolbox.layerNamesIdx][l]
        if not (("flatten" in layerName) or ("pool" in layerName) or ("input" in layerName)):
            allFilters = np.arange(len(serviceOutput[visualizationToolbox.dataStartingIdx+0]))
            useless = np.setdiff1d(allFilters, representants[l])
            result.append(useless.tolist())

    return result


def representativeStatisticsToSpecificFile(visualizationToolbox, representivesStats, filePath):
    output = filePath
    if os.path.exists(output):
        os.remove(output)
    outputLayers = visualizationToolbox.defaultLayerNames
    # outputLayers = [layerName for layerName in outputLayers if not (("flatten" in layerName) or ("input" in layerName) or ("pool" in layerName) or ("dense" in layerName))]
    outputLayers = [layerName for layerName in outputLayers if not (("flatten" in layerName) or ("input" in layerName))]
    with open(output, "a") as textFile:
        for inputIdx in range(len(representivesStats)):
            for layerIdx in range(len(representivesStats[inputIdx])):
                s = "Example: " + str(inputIdx)
                s += " Layer: " + outputLayers[layerIdx]
                s += " Representatives: " + str(representivesStats[inputIdx][layerIdx])
                s += "\n"
                textFile.write(s)


def reverseFilterList(visualizationToolbox, filterObject):
    serviceOutput = visualizationToolbox.loadData(dataset=Dataset.TRAIN.value, fastMode=Modes.MINIMAL.value)
    convIds = [i + visualizationToolbox.dataStartingIdx for i in range(len(serviceOutput[visualizationToolbox.layerNamesIdx])) if "conv" in serviceOutput[visualizationToolbox.layerNamesIdx][i]]
    result = []
    for idx, fList in enumerate(filterObject):
        reversedList = np.setdiff1d(np.arange(len(serviceOutput[convIds[idx]])), fList)
        result.append(reversedList)

    return result


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def getShortName(layerName, layerType):
    baseName = "MaxPool" if layerType == "MaxPooling1D" else layerType
    layerName = baseName.lower() + layerName[layerName.rfind('_'):]
    return layerName


# Normalize the points
def normalizeValues(values):
    minVal = np.min(values)
    maxVal = np.max(values)
    if (maxVal - minVal) == 0:
        values = (values - minVal)
    else:
        values = (values - minVal) / (maxVal - minVal)
    return values


def decrementIfGreaterThan(intList, threshold):
    result = intList
    for i in range(len(result)):
        value = result[i]
        result[i] = value-1 if value > threshold else value
    return result


def joinInnerLists(list1, list2):
    longer, shorter = (list1, list2) if len(list1) > len(list2) else (list2, list1)
    longer = copy.deepcopy(longer)
    shorter = copy.deepcopy(shorter)
    result = longer
    for i, sublistSmall in enumerate(shorter):
        for element in sublistSmall:
            sublistLong = result[i]
            if not element in sublistLong:
                sublistLong.append(element)

    return result


def maybeDelete(filepath):
    try:
        os.remove(filepath)
    except:
        pass


def ensureDirExists(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def percentChange(oldVal, newVal):
    if newVal == oldVal:
        out = 0.0
    else:
        out = ((newVal - oldVal) / float(oldVal)) * 100.0
    return out


def rectifyMetrics(visualizationToolbox, datasetType, score, scoreP):
    metricNames = visualizationToolbox.model.metrics_names
    if datasetType == Dataset.REGRESSION:
        newMetricNames = []
        newScore = []
        newScoreP = []
        for metricIdx, metric in enumerate(metricNames):
            if metric == "acc":
                continue

            if metric == "loss":
                newMetricNames.append("MSE")
            elif metric == "mean_absolute_error":
                newMetricNames.append("MAE")
            else:
                newMetricNames.append(metric)

            # Append the new scores
            newScore.append(score[metricIdx])
            newScoreP.append(scoreP[metricIdx])

        # Replace the old scores with the new scores
        score = newScore
        scoreP = newScoreP
        metricNames = newMetricNames

    return [metricNames, score, scoreP]


def getLayerType(layerName):
    layerName = layerName.lower()

    if "input" in layerName:
        return "Input"
    elif "conv" in layerName:
        return "Conv"
    elif "pool" in layerName:
        return "Pool"
    elif "act" in layerName or "relu" in layerName or "sigmoid" in layerName or "tanh" in layerName or "leaky" in layerName:
        return "Activation"
    elif "dense" in layerName or "fully" in layerName or "fc" in layerName:
        return "Dense"
    elif "norm" in layerName or "bn" in layerName or "batch" in layerName:
        return "BatchNorm"
    else:
        return "Unknown"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~ Importance values~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def computeAndSaveImportanceValues(visualizationToolbox, filePath, n_examples, valueIdx, dataset):
    # computes Min Max and Mean Importance Values for every filter using n_examples test examples. n_examples should be between 0 and 3950
    importanceValues = computeAllImportanceValuesRandom(visualizationToolbox, n_examples, valueIdx, dataset)
    minMaxMean = computeMinMaxMean(importanceValues)
    np.save(filePath, minMaxMean)
    importanceStatisticsToSpecificFile(visualizationToolbox, minMaxMean, filePath.replace(".npy",".txt"))


def computeRepresentativeValues(visualizationToolbox, filePath, n_examples, valueIdx, dataset):
    representatives = computeAllRepresentativesRandom(visualizationToolbox, n_examples, valueIdx, dataset)
    np.save(filePath, representatives)
    representativeStatisticsToSpecificFile(visualizationToolbox, representatives, filePath.replace(".npy",".txt"))


def parseLine(line):
    """
    line: str line from an importance statistics file
    return: importance values for one filter

    method to parse a line of an importance statistics file. An importance statistics file saves min max mean values for every filter of
    a model. It is create by the method compute minMaxMeanImportanceValues
    """
    i = 0
    while i < len(line):
        if line[i:i+6] == "Layer:":
            j = i+7
            while line[j] != " ":
                j += 1
            name = line[i+7:j]
            i = j
        elif line[i:i+4] == "Min:":
            j = i+5
            while line[j] != " ":
                j += 1
            min_val = float(line[i+5:j])
            i = j
        elif line[i:i+4] == "Max:":
            j = i+5
            while line[j] != " ":
                j += 1
            max_val = float(line[i+5:j])
            i = j
        elif line[i:i+5] == "Mean:":
            j = i+6
            while j != len(line) and line[j] != " ":
                j += 1
            mean_val = float(line[i+5:j])
            i = j
        i += 1

    return name, min_val, max_val, mean_val


def computeAllImportanceValuesRandom(visualizationToolbox, examples, valueIdx, dataset):
    inputIteratorBefore = visualizationToolbox.inputIterator
    if dataset == Dataset.TEST:
        maxLen= visualizationToolbox.testX.shape[0]
    else:
        maxLen= visualizationToolbox.trainX.shape[0]

    assert examples <= maxLen
    trainingExamples = random.sample(range(maxLen), examples)

    # Get correct shapes
    layers = visualizationToolbox.model.layers
    layers = [layer for layer in layers if not ("flatten" in layer.name or "input" in layer.name or "dense" in layer.name)]
    resultList = np.zeros(len(layers), dtype=object)

    for l in range(len(layers)):
        numFilters = layers[l].output_shape[2]
        resultList[l] = np.zeros(numFilters, dtype=object)
        for f in range (numFilters):
            resultList[l][f] = np.zeros(examples)

    for i, j in enumerate(tqdm(trainingExamples)):
        # print("Computing influence of example:", j)
        visualizationToolbox.inputIterator = j
        serviceOutput = visualizationToolbox.loadData(dataset=dataset, fastMode=Modes.MINIMAL.value)
        for l in range(visualizationToolbox.dataStartingIdx, len(serviceOutput)-1):
            for f in range(len(serviceOutput[l])):
                resultList[l-visualizationToolbox.dataStartingIdx][f][i] = serviceOutput[l][f][valueIdx]

    visualizationToolbox.inputIterator = inputIteratorBefore
    return resultList


def computeMinMaxMean(importanceValues):
    resultList = np.zeros(len(importanceValues), dtype=object)
    for l in range(len(importanceValues)):
        numFilters = len(importanceValues[l])
        resultList[l] = np.zeros(numFilters, dtype=object)
        for i in range(numFilters):
            minMaxMean = np.zeros(3)
            minV = np.min(np.asanyarray(importanceValues[l][i]))
            maxV = np.max(np.asanyarray(importanceValues[l][i]))
            meanV = np.mean(np.asanyarray(importanceValues[l][i]))
            minMaxMean[0] = minV
            minMaxMean[1] = maxV
            minMaxMean[2] = meanV
            resultList[l][i] = minMaxMean

    return resultList


def importanceStatisticsToSpecificFile(visualizationToolbox, importanceStats, filePath):
    output = filePath
    if os.path.exists(output):
        os.remove(output)

    with open(output, "a") as textFile:
        for l in range(importanceStats.shape[0]):
            for f in range(importanceStats[l].shape[0]):
                if "input" in visualizationToolbox.defaultLayerNames[0]:
                    layerNames = visualizationToolbox.defaultLayerNames[1:]
                else:
                    layerNames = visualizationToolbox.defaultLayerNames

                s = "Layer: " + layerNames[l]
                s += " Filter: " + str(f)
                s += " Min: " + str(importanceStats[l][f][0])
                s += " Max: "  + str(importanceStats[l][f][1])
                s += " Mean: "  + str(importanceStats[l][f][2])
                s += "\n"
                print(s)
                textFile.write(s)


def getAdjustedFilterIndices(name):
    filepath = os.path.join(".", os.path.join("adjustedFilters", name))
    try:
        with open(filepath, 'r') as file:
            indiceString = file.read()
    except:
        print("could not read adjusted filter file")
        return None

    result = indiceStringToList(indiceString)
    return result


def changeImportanceAdjustedFilters(serviceOutput, adjustedIndices):
    layersWithNames = zip(serviceOutput[2:], serviceOutput[0])
    convLayers = [layerWithName[0] for layerWithName in layersWithNames if "conv" in layerWithName[1]]

    layerIdx = 0
    for convLayer, adjustedIndicesInLayer in zip(convLayers, adjustedIndices):
        print("Set importance to 0 for Filters: " + str(adjustedIndicesInLayer) + " from convolutional Layer " + str(layerIdx))
        for i in range(len(adjustedIndicesInLayer)):
            adjustedFilterIndex = adjustedIndicesInLayer[i]
            convLayer[adjustedFilterIndex][2] = 0
            convLayer[adjustedFilterIndex][6] = [0, 0, 0]
        layerIdx += 1


# appends the min, max and mean importance Values for every filter to the ServiceOutput
def appendFilterImportances(output, importanceValues):
    for layerId in range(2, len(output)-1): # Don't include the last layer
        for i in range(len(output[layerId])):
            output[layerId][i].append(importanceValues[layerId-2][i])


def computeAllRepresentativesRandom(visualizationToolbox, examples, valueIdx, dataset):
    inputIteratorBefore = visualizationToolbox.inputIterator
    if dataset == Dataset.TEST.value:
        maxLen = visualizationToolbox.testX.shape[0]
    else:
        maxLen = visualizationToolbox.trainX.shape[0]
    assert examples <= maxLen
    trainingExamples = random.sample(range(maxLen), examples)

    allRepresentives = np.zeros(examples, dtype=object)
    for i, j in enumerate(tqdm(trainingExamples)):
        # print("Compute Example: " + str(i))
        visualizationToolbox.inputIterator = j
        serviceOutput = visualizationToolbox.loadData(dataset=dataset, fastMode=Modes.PARTIAL.value)
        allRepresentives[i] = computeRepresentives(visualizationToolbox, serviceOutput, valueIdx)

    visualizationToolbox.inputIterator = inputIteratorBefore
    return allRepresentives


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~ Filter Clustering ~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def size_cond(size):
    n = size
    r = 2
    f = math.factorial
    return int(f(n) / (f(r) * f(n-r)))


def compute_wcss(linkage_matrix, series, euclidean_distance=False, verbose=False):
    wcss = []
    cut = np.array(cut_tree(linkage_matrix))
    for num_clusters in range(cut.shape[1]):
        current_cut = cut[:, num_clusters]

        num_clusters = cut.shape[1] - num_clusters
        if verbose:
            print("Selecting %d as number of clusters" % (num_clusters))

        if len(np.unique(current_cut)) != num_clusters:
            print("Error: Values not equal [", np.unique(current_cut), num_clusters, "]")
            exit(-1)

        current_wcss = 0.0
        if num_clusters != cut.shape[1]:  # If every point is not assigned a separate cluster (WCSS = 0.0)
            for cluster in range(num_clusters):
                # Get all points with that particular cluster
                matches = current_cut == cluster
                points = series[matches]
                if verbose:
                    print("Number of clusters:", num_clusters, "| Matches:", matches, "| Points:", points.shape)

                if len(points) > 1:
                    if verbose:
                        print("Computing distance matrix")

                    if euclidean_distance:
                        dist_mat = pdist(points)
                        current_wcss += np.mean(dist_mat)
                    else:
                        dist_mat = np.array(dtw.distance_matrix_fast(points))
                        dist_mat = np.triu(dist_mat, k=1)
                        current_wcss_sum = np.sum(dist_mat)
                        n = dist_mat.shape[0]
                        normalizer = ((n * (n - 1)) / 2.0)
                        if verbose:
                            print("Normalizer:", normalizer)
                        current_wcss += (current_wcss_sum / float(normalizer))
                    if verbose:
                        print("Distance matrix:", dist_mat)
                    # dtw_visualisation.plot_matrix(dist_mat, shownumbers=True)

            current_wcss = current_wcss
        wcss.append(current_wcss)

    return wcss, cut


def normalizeValues(y, scaleXAxis=SCALE_X_AXIS, scaleYAxis=True, negScale=False):
    seqLen = y.shape[0]
    x = np.arange(seqLen)
    if scaleYAxis:
        currentMin = np.min(y)
        currentMax = np.max(y)
        if currentMin != currentMax:
            y = (y - currentMin) / (currentMax - currentMin)
        else:
            y = y - currentMin

        if negScale:  # Have both positive as well as negative values
            y = y - 0.5

    if scaleXAxis:
        nonZeroInd = np.nonzero(y)[0]
        if len(nonZeroInd) == 0:
            return y

        start = max(0, int(nonZeroInd[0]) - 1)
        end = min(int(nonZeroInd[-1]) + 2, seqLen)
        xNonZero = x[start:end]
        yNonZero = y[start:end]
        step = (end - start) / seqLen
        xComp = np.arange(start, end, step=step)
        yComp = np.interp(xComp, xNonZero, yNonZero)
        y = yComp

    return y


def plotVals(numFilters, data, interestPoints, outputPath, plotWCSS=True):
    # Create the plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of clusters')

    # Plot the optimal number of clusters
    if plotWCSS:
        ax.set_title('Elbow Plot', color='C0')
        x = np.arange(1, numFilters + 1)

        distances, wcss = data

        ax.plot(x, wcss, 'C0', label='WCSS', linewidth=2.0)
        ax.plot(x, distances, 'C1', label='Distances', linewidth=2.0)

        ax.plot([x[0], x[-1]], [wcss[0], wcss[-1]], 'C2', label='Line', linewidth=1.0, linestyle=":")
        ax.plot(interestPoints + 1, wcss[interestPoints], 'C3', label='Within cluster SSE', linewidth=0.0, marker="o")
        ax.plot(interestPoints + 1, distances[interestPoints], 'C4', label='Distance to the line', linewidth=0.0, marker="*")

        ax.set_ylabel('Distance')

    else:  # Plot silhouette scores
        ax.set_title('Silhouette Plot', color='C0')
        x = np.arange(2, numFilters)  # Silhouette score cannot be computed for the first two and the last clusters
        silhouette = data

        # Make the interest points 1 indexed (zero shouldn't be included)
        ax.plot(x, silhouette, 'C0', label='Silhouette score', linewidth=2.0)
        ax.plot(interestPoints + 1, silhouette[interestPoints - 1], 'C3', label='Optimal', linewidth=0.0, marker="o")

        ax.set_ylabel('Score')

    ax.legend()
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300)


def asankaFindK(wcss, numFilters, layerOutputDir, visualizeFilters, useGradientInversion=False, verbose=False):
    # Sketch out the imaginary line
    distances = []
    for i in range(0, numFilters):
        p1 = Point(initx=1, inity=wcss[0])
        p2 = Point(initx=numFilters, inity=wcss[-1])
        p = Point(initx=i + 1, inity=wcss[i])
        distances.append(p.distance_to_line(p1, p2))

    if useGradientInversion:
        # Determine point where the gradient inverts sign
        interestPoints = []
        for i in range(1, len(distances) - 1):  # Exclude the first and the last point
            # Updated the gradient inversion points computation (pos-neg)
            # if np.sign(distances[i] - distances[i - 1]) != np.sign(distances[i + 1] - distances[i]):
            if (np.sign(distances[i] - distances[i - 1]) > 0.0) and (np.sign(distances[i + 1] - distances[i]) < 0.0):
                interestPoints.append(i)
    else:
        interestPoints = [np.argmax(distances)]

    # Convert to numpy arrays
    wcss = np.array(wcss)
    distances = np.array(distances)
    interestPoints = np.array(interestPoints)

    if verbose:
        print("Interest points:", interestPoints)

    if visualizeFilters:
        outputPath = os.path.join(layerOutputDir, SELECTION_TYPE + ".png")
        plotVals(numFilters, [distances, wcss], interestPoints, outputPath, plotWCSS=True)

    numClusters = interestPoints[0]  # Choose the first gradient inversion point

    return numClusters


def silhouetteFindK(cut, dists, layerOutputDir, visualizeFilters, verbose=False):
    # Complete the dist matrix
    for i in range(dists.shape[0]):
        for j in range(i, dists.shape[1]):
            if i == j:
                dists[i, j] = 0.0
            else:
                dists[j, i] = dists[i, j]

    silhouetteScores = []
    for num_clusters in range(1, cut.shape[1] - 1):  # Exclude the first and the last two clusters. Valid labels: [2, N-2]
        current_cut = cut[:, num_clusters]
        score = sklearn.metrics.silhouette_score(dists, current_cut, metric='precomputed')  # [-1, +1]
        silhouetteScores.append(score)

    silhouetteScores = np.array(silhouetteScores)
    assert not np.isnan(silhouetteScores).any(), "Error: NaN values occured in the silhouette score. Please use the latest version of scikit-learn to avoid this error!"
    numClusters = np.argmax(silhouetteScores) + 1  # Compensate for the fact that the cut started from one

    if verbose:
        print("Scores:", silhouetteScores)
        print("Num clusters:", numClusters)

    if visualizeFilters:
        numFilters = cut.shape[1]
        outputPath = os.path.join(layerOutputDir, SELECTION_TYPE + ".png")
        plotVals(numFilters, silhouetteScores[::-1], numFilters - numClusters - 1, outputPath, plotWCSS=False)

    return numClusters


# RawFilterValues corresponds to outputs of just a single layer
def computeFilterClusters(visualizationToolbox, serviceOutput, visualizeFilters=False, verbose=False):
    if visualizeFilters:
        outputDir = './OutputPlots'
        if os.path.exists(outputDir):
            shutil.rmtree(outputDir)
        os.mkdir(outputDir)

    skipFurtherLayers = False
    for layerId in range(visualizationToolbox.dataStartingIdx, len(serviceOutput)):  # Don't include the last layer
        layerName = serviceOutput[visualizationToolbox.layerNamesIdx][layerId - visualizationToolbox.dataStartingIdx]
        if verbose:
            print("Layer Name:", layerName)

        # Disregard the dense layers
        if "dense" in layerName:
            skipFurtherLayers = True

        if not skipFurtherLayers:
            layerOutputDir = None
            if visualizeFilters:
                layerOutputDir = os.path.join(outputDir, layerName)
                os.mkdir(layerOutputDir)

            # Create clusters for the raw feature values
            output = []
            numFilters = len(serviceOutput[layerId])
            for i in range(numFilters):
                rawValOut = np.array(serviceOutput[layerId][i][visualizationToolbox.rawValueIdx])

                # Make this series scale invariant for clustering (min-max scaling)
                rawValOut = normalizeValues(rawValOut, negScale=False)
                output.append(rawValOut)

            rawFilterValues = np.array(output)
            assert rawFilterValues.shape[0] == numFilters
            if verbose:
                print("Layer name: %s | Output shape: %s" % (serviceOutput[visualizationToolbox.layerNamesIdx][layerId - visualizationToolbox.dataStartingIdx], str(rawFilterValues.shape)))

            if CLUSTERING_METHOD == Clustering.K_MEANS:  # K-Means
                numClusters = int(numFilters / 5)  # Randomly start with this quantity
                kmeans = KMeans(n_clusters=numClusters, random_state=RANDOM_STATE).fit(rawFilterValues)
                labels = kmeans.labels_

            elif CLUSTERING_METHOD == Clustering.ADAPTIVE_K_MEANS:  # Adaptive K-Means
                wcss = []
                for i in range(1, numFilters + 1):
                    kmeans = KMeans(n_clusters=i, random_state=RANDOM_STATE).fit(rawFilterValues)
                    wcss.append(kmeans.inertia_)

                # Compute optimal number of clusters based on Asanka algorithm
                numClusters = asankaFindK(wcss, numFilters, layerOutputDir, visualizeFilters)

                # Train the K-Means classifier with optimal number of clusters
                kmeans = KMeans(n_clusters=numClusters, random_state=RANDOM_STATE).fit(rawFilterValues)

                labels = kmeans.labels_

            elif CLUSTERING_METHOD == Clustering.GMM:  # GMM
                numClusters = int(numFilters / 5)  # Randomly start with this quantity
                gmm = sklearn.mixture.BayesianGaussianMixture(n_components=numClusters)
                gmm.fit(rawFilterValues)
                labels = gmm.predict(rawFilterValues)

            elif CLUSTERING_METHOD == Clustering.MEAN_SHIFT:  # Mean-shift
                meanShift = MeanShift(bandwidth=MEAN_SHIFT_BANDWIDTH)
                labels = meanShift.fit_predict(rawFilterValues)

            elif CLUSTERING_METHOD == Clustering.HIERARCHICAL:  # Hierarchical clustering
                dists = dtw.distance_matrix_fast(rawFilterValues)
                assert dists is not None, "DTAI distance library returned None! Fast version not available."
                dists_cond = np.zeros(size_cond(rawFilterValues.shape[0]))

                idx = 0
                for r in range(rawFilterValues.shape[0] - 1):
                    dists_cond[idx:idx + rawFilterValues.shape[0] - r - 1] = dists[r, r + 1:]
                    idx += rawFilterValues.shape[0] - r - 1

                # Compute the linkage matrix
                z = linkage(dists_cond, method='complete', metric='euclidean')

                if verbose:
                    print("Distance matrix:\n{}".format(dists))
                    print("Dists shape:", dists_cond.shape, "| Filters:", rawFilterValues.shape[0])
                    print("Z:", z)

                if visualizeFilters:
                    plt.title('Hierarchical Clustering Dendrogram')
                    plt.xlabel('Filter index')
                    plt.ylabel('Distance')
                    dendrogram(z, show_leaf_counts=True, leaf_rotation=90., leaf_font_size=6, show_contracted=True)
                    plt.tight_layout()

                    outputPath = os.path.join(layerOutputDir, "dendrogram.png")
                    plt.savefig(outputPath, dpi=300)

                if SELECTION_TYPE == Clustering.ASANKA:  # Asanka
                    wcss, cut = compute_wcss(z, rawFilterValues)  # Invert the list so that it starts from 1 to n
                    wcss = wcss[::-1]

                    # Compute optimal number of clusters based on Asanka algorithm
                    numClusters = asankaFindK(wcss, numFilters, layerOutputDir, visualizeFilters, verbose=verbose)
                    numClusters = cut.shape[1] - numClusters

                elif SELECTION_TYPE == Clustering.SILHOUETTE:  # Silhouette
                    cut = np.array(cut_tree(z))
                    numClusters = silhouetteFindK(cut, dists, layerOutputDir, visualizeFilters, verbose=verbose)

                else:
                    print("Error: Unknown cluster selection method (%s)" % SELECTION_TYPE)
                    exit(-1)

                # Get predictions with optimal number of clusters
                labels = cut[:, numClusters]

            else:
                print("Error: Unknown clustering method (%s)" % CLUSTERING_METHOD)
                exit(-1)

            # Pick the cluster centers and their representative set
            assignedClusters = np.unique(labels)

            if verbose:
                print("Assigned clusters:", assignedClusters)

            if visualizeFilters:
                # Write the clusters into one directory
                for i in range(numFilters):
                    currentPlotOutputDir = os.path.join(layerOutputDir, str(labels[i]))
                    if not os.path.exists(currentPlotOutputDir):
                        os.mkdir(currentPlotOutputDir)
                    outputPath = os.path.join(currentPlotOutputDir, str(i) + ".png")
                    print("Creating plot: %s" % outputPath)

                    # Create the plot
                    x = np.arange(1, rawFilterValues.shape[1] + 1)

                    fig, ax = plt.subplots()
                    ax.set_title('Filter # ' + str(i), color='C0')

                    ax.plot(x, rawFilterValues[i, :], 'C0', label='Raw Output', linewidth=2.0)
                    ax.legend()

                    ax.set_xlabel('Time-step')
                    ax.set_ylabel('Raw value')

                    plt.tight_layout()

                    plt.savefig(outputPath, dpi=300)
                    # plt.show()
                    plt.close('all')
            else:
                for i in range(len(labels)):
                    serviceOutput[layerId][i].append(int(labels[i]))

        # No clustering can be performed for dense layers
        else:
            for i in range(len(serviceOutput[layerId])):
                serviceOutput[layerId][i].append(-1)
