#!flask/bin/python

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import matplotlib.pyplot as plt

from tsviz import VisualizationToolbox
from utils import *

try:
    import cPickle as pickle  # Python2
except ImportError:
    import pickle  # Python3

# Initialize the model
print("Initializing the model")
visualizationToolbox = VisualizationToolbox()
visualizationToolbox.initializeTensorflow()

# Spawn the service
print("Initializing the service")
app = Flask(__name__)
# cors = CORS(app)


@app.route('/viz/api/train', methods=['GET'])
def switchToTrain():
    if visualizationToolbox.switchToTrain():
        return jsonify({'status': 'ok'})


@app.route('/viz/api/test', methods=['GET'])
def switchToTest():
    if visualizationToolbox.switchToTest():
        return jsonify({'status': 'ok'})


@app.route('/viz/api/reset', methods=['GET'])
def resetIterator():
    if visualizationToolbox.resetIterator():
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'Unable to reset iterator'})


@app.route('/viz/api/set', methods=['GET'])
def setIterator():
    if 'iterator' in request.args:
        iterator = int(request.args.get('iterator', 0))
        success = visualizationToolbox.setIterator(iterator)
        output = {'status': 'ok'} if success else {'status': 'error', 'msg': 'Iterator exceeded the maximum possible value'}
        return jsonify(output)
    else:
        return jsonify({'status': 'error', 'msg': 'Iterator value not found'})


@app.route('/viz/api/prev', methods=['GET'])
def loadPreviousInput():
    if visualizationToolbox.prevInputButton():
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'No previous examples in the dataset'})


@app.route('/viz/api/next', methods=['GET'])
def loadNextInput():
    if visualizationToolbox.nextInputButton():
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'No more examples in the dataset'})


@app.route('/viz/api/get_iterator', methods=['GET'])
def getIterator():
    it = visualizationToolbox.getIterator()
    return jsonify({'status': 'ok', 'iterator': it})


@app.route('/viz/api/get_example', methods=['GET'])
def getExample():
    X, y = visualizationToolbox.getExample()
    return jsonify({'status': 'ok', 'x': X.T.tolist(), 'y': y.tolist()})


@app.route('/viz/api/get_prediction', methods=['GET'])
def getPrediction():
    X, y, pred, saliency = visualizationToolbox.getPrediction()
    return jsonify({'status': 'ok', 'x': X.T.tolist(), 'y': y.tolist(), 'pred': pred.tolist(), 'saliency': saliency.tolist()})


@app.route('/viz/api/get_architecture', methods=['GET'])
def getArchitecture():
    arch = visualizationToolbox.getArchitecture()
    return jsonify({'status': 'ok', 'architecture': arch})


@app.route('/viz/api/set_percentile', methods=['GET'])
def setPercentile():
    if 'percentile' in request.args:
        percentile = float(request.args.get('percentile', -1))
        visualizationToolbox.changePercentile(percentile)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'Percentile value not found'})


@app.route('/viz/api/fetch', methods=['GET'])
def fetchData():
    return jsonify({'data': visualizationToolbox.loadData()})


@app.route("/viz/api/classify", methods=['GET'])
def classify():
    dataShape = visualizationToolbox.trainX.shape[1:]
    seq = ";".join([",".join(['0' for i in range(dataShape[0])]) for j in range(dataShape[1])])

    if 'seq' in request.args:
        seq = request.args.get('seq', seq)

    seq = seq.split(";")
    seq = np.array([list(eval(x)) for x in seq]).T
    return jsonify({'status': 'ok', 'prediction': str(visualizationToolbox.classifySequence(seq))})


@app.route('/viz/api/cluster', methods=['GET'])
def performClustering():
    visualizationToolbox.performFilterClustering()
    return jsonify({'status': 'ok'})


@app.route('/viz/api/visualize_clusters', methods=['GET'])
def visualizeClusters():
    visualizationToolbox.visualizeFilterClusters()
    return jsonify({'status': 'ok'})


@app.route('/viz/api/prune', methods=['GET'])
def performPruning():
    if 'indices' in request.args and 'name' in request.args and 'epochs' in request.args and 'mode' in request.args:
        epochs = int(request.args.get('epochs', 1))
        indices = indiceStringToList(request.args.get('indices', "0;;"))
        name = request.args.get('name', "my_model")
        mode = request.args.get('mode', Modes.PRUNE.value)
        '''
        # if name = "", create timestamp
        if name == "":
            name = datetime.now().strftime('%Y%m%d%H%M%S')
        else: 
            name += datetime.now().strftime('%Y%m%d%H%M%S')
        '''
        visualizationToolbox.performNetworkPruning(epochs, indices, name, mode)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'status': 'error', 'msg': 'Indices value, name value, mode value or epochs value not found'})


@app.route("/viz/api/load_pruned_model", methods=["GET"])
def loadPrunedModel():
    if "name" in request.args:
        name = request.args.get("name", "my_model")
        path = os.path.join(".", os.path.join("prunedModels", name))
        visualizationToolbox.modelPath = path + ".h5"
        visualizationToolbox.currentlyLoadedModel = name
        visualizationToolbox.loadModel()
        # TODO: Compute the importance over number of examples
        #visualizationToolbox.computeMinMaxMeanImportanceValues(name, n_examples=100)
        #visualizationToolbox.loadImportanceValues(name)
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "error", "msg": "name argument could not be found"})    


@app.route("/viz/api/test_model", methods=["GET"])
def testCurrentModel():
    result = visualizationToolbox.performTesting()
    return jsonify({'status': 'ok', 'Result': result})


# Returns the pruned model names that are in use.
@app.route("/viz/api/get_used_model_names", methods=['GET'])
def getUsedModelNames():
    names = []
    try:
        file = open(os.path.join('.', os.path.join('prunedModels', 'prunedModelNames')))
        for name in file:
            name = name.replace(os.linesep, '')
            if name == '':
                continue
            names.append(name)
        file.close()
    except:
        pass

    return jsonify({'names': names})


# Delete pruned model and its importance file
@app.route("/viz/api/delete_pruned_model", methods=['GET'])
def deletePrunedModel():
    if 'name' in request.args:
        name = request.args.get("name", "my_model")
        dirpath = os.path.dirname(os.path.realpath(__file__))
        dirpath = os.path.join(dirpath, "prunedModels")
        filepath = os.path.join(dirpath, "prunedModelNames")

        # erase the model name from file that saves the model names that are in use.
        linesToKeep = []
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    if line == name + os.linesep:
                        continue
                    linesToKeep.append(line)
        except:
            pass

        with open(filepath, 'w') as file:
            file.writelines(linesToKeep)

        # Delete min, max, mean importance statistics file
        importanceFilepath = os.path.join(".", os.path.join("ImportanceStatistics", name))
        maybeDelete(importanceFilepath)

        # Delete the actual file the model is saved in (not actually necessary)
        modelFilepath = os.path.join(".", os.path.join("prunedModels", name + ".h5"))
        maybeDelete(modelFilepath)

        # Delete adjusted filter information file of the model (not actually necessary)
        adjustedFilepath = os.path.join(".", os.path.join("adjustedFilters", name))
        maybeDelete(adjustedFilepath)

        return jsonify({'status': 'ok'})

    else:
        return jsonify({'status': 'error', 'msg': 'name argument could not be found'})


@app.route("/viz/api/load_default_model", methods=['GET'])
def loadDefaultModel():
    visualizationToolbox.modelPath = visualizationToolbox.standardModelPath
    visualizationToolbox.currentlyLoadedModel = visualizationToolbox.standardModelName
    visualizationToolbox.loadModel()
    # TODO: Importance values for number of examples
    #visualizationToolbox.loadImportanceValues(visualizationToolbox.standardModelName)
    return jsonify({'status': 'ok'})


@app.route("/viz/api/get_loaded_model_name", methods=['GET'])
def getLoadedModelName():
    return jsonify({'status': 'ok', 'name': visualizationToolbox.currentlyLoadedModel})


# Checks if name is already used by another model
@app.route("/viz/api/name_available", methods=['GET'])
def checkNameAvailable():
    if 'name' in request.args:
        dirpath = os.path.dirname(os.path.realpath(__file__))
        dirpath = os.path.join(dirpath, "prunedModels")
        filepath = os.path.join(dirpath, "prunedModelNames")
        name = request.args.get("name", "my_model")

        # Go through file that stores the already used names. And checks if name is in there or not.
        if not name == '':
            try:
                with open(filepath, 'r') as file:
                    available = 'True'
                    for used_name in file:
                        used_name = used_name.replace(os.linesep, '')
                        if name == used_name:
                            available = 'False'
                            break
            except:
                available = 'True'
        else:
            available = 'False'

        return jsonify({'status': 'ok', 'available': available})

    else:
        return jsonify({'status': "error"})


# Returns an alternative to the name argument in the url. The alternative is the "name" argument and the smallest number possible attached to it, so the alternative name is available
@app.route("/viz/api/get_alternative_to_name", methods=['GET'])
def getAlternativeToName():
    if 'name' in request.args:
        dirpath = os.path.dirname(os.path.realpath(__file__))
        dirpath = os.path.join(dirpath, "prunedModels")
        filepath = os.path.join(dirpath, "prunedModelNames")
        alternative = request.args.get("name", "my_model")
        i = 0
        try:
            with open(filepath, 'r') as file:
                while alternative + str(i) + os.linesep in file:
                    i += 1
        except:
            pass
        alternative += str(i)
        return jsonify({'status': 'ok', 'alternative': alternative})
    else:
        return jsonify({'status': "error"})


@app.route("/viz/api/get_filter_list", methods=['GET'])
def getFilterList():
    #if datamode==1 ? compute for all examples : compute for selected
    dataMode = int(request.args.get("data_mode", Modes.SELECTED_EXAMPLES.value))

    # if importance_mode==0 ? percentile maximum
    # if importance_mode==1 ? percentile minimum
    # if importance_mode==2 ? sorted importance maximum
    # if importance_mode==3 ? sorted importance miminum
    # if importance_mode==4 ? cluster representatives
    importanceMode = int(request.args.get("importance_mode", Modes.PERCENTILE_MAXIMUM.value))

    # if importance_mode==(0 or 1) ? percentile value
    # if importance_mode==(2 or 3) ? number of top most results
    numberOfFilter = int(request.args.get("number_of_filter", 0))

    # list of layerids included in the process e.g. network =[conv,max,conv,max] -> 0,1 to process both conv layers
    # the number is not the actual layer, it is the number of the convlayer.
    # first conv layer in the network is 0 and second is 1 (does not depend on other layers between those two cons layers)
    layerSelection = np.fromstring(request.args.get("layer_selection", "0"), dtype=int, sep=",")

    # if importanceSelection==0 ? importance : loss
    importanceSelection = int(request.args.get("importanceSelection", Modes.IMPORTANCE.value))

    # if dataset==1 ? testset : trainset
    dataset = int(request.args.get("dataset", Dataset.TRAIN.value))

    result = visualizationToolbox.computePruningFilterSet(dataMode, importanceMode, numberOfFilter, layerSelection, importanceSelection, dataset)
    result = listToIndiceString(result)
    return jsonify({'status': 'ok', 'indices': result})


@app.route('/viz/api/load_custom_dataset', methods=['GET'])
def loadCustomDataset():
    if 'dataPath' in request.args:
        dataPath = request.args.get("dataPath", "./datamark-internettrafficdata.csv")
        if os.path.exists(dataPath):
            if os.path.isfile(dataPath):
                datasetStats = visualizationToolbox.loadCustomSingleChannelDatasets(dataPath)
                return jsonify({'status': 'ok', 'stats': datasetStats})
        return jsonify({'status': 'No valid dataset path provided'})
    return jsonify({'status': 'No dataset provided'})


@app.route("/viz/api/get_filter_list_from_file", methods=['GET'])
def getFilterListFromFile():
    # if mode==0 ? compute random
    # if mode==1 ? compute percentile
    # if mode==2 ? compute representative
    mode = int(request.args.get("mode", Modes.COMPUTE_RANDOM.value))

    # if submode==0 ? min
    # if submode==1 ? max
    # if submode==2 ? mean
    submode = int(request.args.get("submode", Modes.MEAN.value))

    # if mode==(0 or 1) ? percentile
    percentile = int(request.args.get("percentile", 10))

    # if mode==(1 or 2) ? lowest==0 or most important==1
    reverse = int(request.args.get("reverse", 0))

    # if mode==(1 or 2) ? number of examples
    examples = int(request.args.get("examples", 100))

    # if importanceSelection==0 ? importance : loss
    importanceSelection = int(request.args.get("importanceSelection", Modes.IMPORTANCE.value))

    # if dataset==1 ? testset : trainset
    dataset = int(request.args.get("dataset", Dataset.TRAIN.value))

    result = visualizationToolbox.computePruningFilterSetFromFile(mode, submode, percentile, reverse, examples, importanceSelection, dataset)
    result = listToIndiceString(result)
    return jsonify({'status': 'ok', 'indices': result})


# This method computes all pruning results reported in the paper
@app.route("/viz/api/compute_all_pruning_results", methods=['GET'])
def getPruningResults():
    dataset = Dataset.TRAIN.value
    examples = -1  # Complete dataset
    reverse = 0
    mode = Modes.COMPUTE_PERCENTILE.value
    percentile = 10  # Doesn't matter
    submode = Modes.MEAN.value
    epochs = 10
    pruningMode = Modes.PRUNE.value
    adjustMode = Modes.ADJUST.value
    visualizationToolbox.setIterator(0)  # Set the iterator to the first example

    lossModeDisabled = True

    # Compute the min, mean and max importance for each filter for the complete dataset
    modelName = visualizationToolbox.currentlyLoadedModel
    dirPath = os.path.join(".", "Statistics")
    dirPath = os.path.join(dirPath, modelName)
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)

    convLayersIdx = [layerIdx for layerIdx, layerName in enumerate(visualizationToolbox.getModelLayerNames()) if "conv" in layerName.lower()]
    numConvLayers = sum([1 for layerName in visualizationToolbox.getModelLayerNames() if "conv" in layerName.lower()])
    print("Model layers:", visualizationToolbox.getModelLayerNames())
    print("Number of convolutional layers:", numConvLayers, "| Convolutional layers idx:", convLayersIdx)
    completeResults = []

    faithfulnessResultsFileName = os.path.join(dirPath, "Faithfulness.txt")
    if not os.path.exists(faithfulnessResultsFileName):
        faithfulnessResultsFile = open(faithfulnessResultsFileName, "w")
    else:
        faithfulnessResultsFile = None

    # importance = Modes.IMPORTANCE.value  # 0: Loss, 1: Importance
    for importanceType in range(2):
        # Importance values | 0: Loss, 1: Importance
        if (importanceType == 0) and lossModeDisabled:
            print("Computing statistics based on loss")
            importance = Modes.LOSS.value
            continue
        else:
            print("Computing statistics based on importance")
            importance = Modes.IMPORTANCE.value
            measure = "importance"

        completeResults.append([])

        # measure = "importance" if importance == Modes.IMPORTANCE.value else "loss"
        filePath = os.path.join(dirPath, "MinMaxMean_" + measure + ".npy")

        # Load the default model
        if visualizationToolbox.currentlyLoadedModel != visualizationToolbox.standardModelPath:
            print("Loading default model again!")
            with K.get_session().graph.as_default():
                visualizationToolbox.modelPath = visualizationToolbox.standardModelPath
                visualizationToolbox.currentlyLoadedModel = visualizationToolbox.standardModelName
                visualizationToolbox.loadModel()

        if not os.path.exists(filePath):
            result = visualizationToolbox.computePruningFilterSetFromFile(mode, submode, percentile, reverse, examples, importance, dataset)

        # Load the computed statistics
        importanceStats = np.load(filePath)

        for convLayerIdx in range(numConvLayers):
            # Create the plot
            layerIdx = convLayersIdx[convLayerIdx]  # Convert the conv layer index to actual layers
            selectedLayer = importanceStats[layerIdx]
            layerName = visualizationToolbox.getModelLayerNames()[layerIdx]
            print("Computing statistics for layer:", layerName)

            fileNamePartial = layerName + ("_loss" if importance == Modes.LOSS.value else "_importance")
            pickleFileName = os.path.join(dirPath, fileNamePartial + ".pickle")

            filterMinImp = [x[Modes.MINIMUM.value] for x in selectedLayer]
            filterMaxImp = [x[Modes.MAXIMUM.value] for x in selectedLayer]
            filterMeanImp = [x[Modes.MEAN.value] for x in selectedLayer]

            assert len(filterMinImp) > 1
            assert len(filterMaxImp) > 1
            assert len(filterMeanImp) > 1
            print("Filter statistics length | Minimum importance: %d | Maximum importance: %d | Mean importance: %d" % (len(filterMinImp), len(filterMaxImp), len(filterMeanImp)))

            x = np.arange(0, len(selectedLayer))

            fig, ax = plt.subplots()
            if lossModeDisabled:
                ax.set_title('Filter importance for layer ' + layerName, color='C0')
            else:
                ax.set_title('Filter importance for layer ' + layerName + ' w.r.t. ' + ('loss' if importance == Modes.LOSS.value else 'importance'), color='C0')

            ax.plot(x, filterMinImp, 'C0', label='Minimum', linewidth=1.0, marker='o')
            ax.plot(x, filterMaxImp, 'C1', label='Maximum', linewidth=1.0, marker='o')
            ax.plot(x, filterMeanImp, 'C2', label='Mean', linewidth=1.0, marker='o')

            ax.legend()

            ax.set_xlabel('Filter ID')
            ax.set_ylabel('Importance')

            plt.tight_layout()

            outputPath = os.path.join(dirPath, "plot_" + fileNamePartial + ".png")
            plt.savefig(outputPath, dpi=300)
            plt.close('all')

            # Create the plot for percentile based pruning
            subMode = Modes.MEAN.value  # Performing pruning based on mean importance
            filterRanks = np.argsort(filterMeanImp)  # In ascending order by default going from least to max importance
            results = []

            if not os.path.exists(pickleFileName):
                metricNames = None
                with open(os.path.join(dirPath, "Pruning-complete-" + fileNamePartial + ".txt"), "w") as file:
                    numFilters = len(filterMeanImp)
                    for numFiltersToDelete in range(1, numFilters):  # Remove 1 to numFilters - 1 filters based on least importance
                        # Get the least important filters based on importance
                        filtersToDelete = [[] for idx in range(numConvLayers)]
                        filtersToDelete[convLayerIdx] = filterRanks[:numFiltersToDelete]
                        indices = listToIndiceString(filtersToDelete)
                        print("Pruning %d filters. Filters Idx: %s" % (numFiltersToDelete, indices))
                        file.write("Pruning %d filters. Filters Idx: %s\n" % (numFiltersToDelete, indices))

                        # Use these indices for pruning
                        name = "filters-" + str(numFiltersToDelete) + "-" + fileNamePartial
                        score, scoreP = visualizationToolbox.performNetworkPruning(epochs, filtersToDelete, name, pruningMode)
                        metricNames, score, scoreP = rectifyMetrics(visualizationToolbox, DATASET_TYPE, score, scoreP)

                        if len(results) == 0:
                            results.append(score)
                        results.append(scoreP)

                        for idx in range(len(metricNames)):
                            file.write("%s (%s): %f, %f, %f%%\n" % (metricNames[idx], "loss" if importance == Modes.LOSS.value else "importance", score[idx], scoreP[idx], percentChange(score[idx], scoreP[idx])))

                x = np.arange(0, numFilters)

                # Save x and results in pickle file
                assert metricNames is not None, "Error: Metric names cannot be None"
                with open(pickleFileName, "wb") as file:
                    pickle.dump([x, results, metricNames], file, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                # Load x and results from pickle file
                with open(pickleFileName, "rb") as file:
                    x, results, metricNames = pickle.load(file)

            # Add the results for the final plots
            completeResults[-1].append([x, results, layerIdx, metricNames])

            fig, ax = plt.subplots()
            if lossModeDisabled:
                ax.set_title('Pruning on layer ' + layerName, color='C0')
            else:
                ax.set_title('Pruning on layer ' + layerName + ' w.r.t. ' + ('loss' if importance == Modes.LOSS.value else 'importance'), color='C0')

            for idx in range(len(metricNames)):
                ax.plot(x, [x[idx] for x in results], label=metricNames[idx], linewidth=1.0, marker='o', color='C' + str(idx))

            ax.legend()

            ax.set_xlabel('Filters removed')
            ax.set_ylabel('Metric value')

            plt.tight_layout()

            outputPath = os.path.join(dirPath, "pruning_plot_" + fileNamePartial + ".png")
            plt.savefig(outputPath, dpi=300)
            plt.close('all')

            if faithfulnessResultsFile is not None:
                # Experiments for 'Faithfulness'
                mostInfluentialFilter = filterRanks[-1]
                leastInfluentialFilter = filterRanks[0]

                for currentIndice, filterType in [(leastInfluentialFilter, "Least influential"), (mostInfluentialFilter, "Most influential")]:
                    filtersToDelete = [[]] * numConvLayers
                    filtersToDelete[convLayerIdx] = [currentIndice]
                    filtersToDeleteStr = listToIndiceString(filtersToDelete)
                    print("Pruning filter from layer %s. Filters Idx: %s\n" % (layerName, filtersToDeleteStr))
                    faithfulnessResultsFile.write("Pruning filter from layer %s. Filters Idx: %s\n" % (layerName, filtersToDeleteStr))
                    score, scoreP = visualizationToolbox.performNetworkPruning(-1, filtersToDelete, "", adjustMode)
                    metricNames, score, scoreP = rectifyMetrics(visualizationToolbox, DATASET_TYPE, score, scoreP)

                    for idx in range(len(metricNames)):
                        faithfulnessResultsFile.write("%s (%s) - %s: %f, %f, %f%%\n" % (metricNames[idx], "loss" if importance == Modes.LOSS.value else "importance", filterType, score[idx], scoreP[idx], percentChange(score[idx], scoreP[idx])))

    if faithfulnessResultsFile is not None:
        faithfulnessResultsFile.close()

    # Save the complete results in pickle file
    pickleFileName = os.path.join(dirPath, "complete.pickle")
    with open(pickleFileName, "wb") as file:
        pickle.dump(completeResults, file, protocol=pickle.HIGHEST_PROTOCOL)

    for layerIterator in range(len(completeResults[0])):
        if len(completeResults) == 2:  # If both influences (w.r.t. loss and w.r.t. importance)
            x, resultsLoss, layerIdx, metricNames = completeResults[0][layerIterator]
            _, resultsImportance, _, _ = completeResults[1][layerIterator]
        else:
            x, resultsImportance, layerIdx, metricNames = completeResults[0][layerIterator]

        layerName = visualizationToolbox.getModelLayerNames()[layerIdx]

        fig, ax = plt.subplots()
        ax.set_title('Pruning on layer: ' + layerName, color='C0')

        for idx in range(len(metricNames)):
            if len(completeResults) == 2:  # If both influences (w.r.t. loss and w.r.t. importance)
                ax.plot(x, [x[idx] for x in resultsLoss], label=metricNames[idx] + ' (Loss)', linewidth=1.0, marker='o', color='C' + str(idx))
                ax.plot(x, [x[idx] for x in resultsImportance], label=metricNames[idx] + '(Importance)', linewidth=1.0, marker='o', color='C' + str(idx))
            else:
                ax.plot(x, [x[idx] for x in resultsImportance], label=metricNames[idx], linewidth=1.0, marker='o', color='C' + str(idx))

        ax.legend()

        ax.set_xlabel('Filters removed')
        ax.set_ylabel('Metric value')

        plt.tight_layout()

        outputPath = os.path.join(dirPath, "pruning_plot_" + layerName + ".png")
        plt.savefig(outputPath, dpi=300)
        plt.close('all')

    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    # Create the required directories
    ensureDirExists("./prunedModels")
    ensureDirExists("./adjustedFilters")
    ensureDirExists("./Statistics")

    # Start the service
    print("Starting the service")
    app.run(host="0.0.0.0", port=5000)
