#!/bin/python

import os

# Packages for DL
import tensorflow as tf
import keras.backend as K
from keras.models import load_model, save_model
from keras.backend.tensorflow_backend import set_session
import pandas as pd
import numpy as np
import scipy.stats
import math
from tqdm import tqdm

from keras import regularizers, optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D, Flatten, Dense, Lambda, Conv2DTranspose, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_style("white")
from utils import *
from config import *

try:
    import cPickle as pickle  # Python2
except ImportError:
    import pickle  # Python3

# Ignore keras 2 API warnings
import warnings
warnings.filterwarnings("ignore")

# # Create a new session for keras
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))


class VisualizationToolbox:
    def __init__(self):
        # Define the API data indices
        self.datasetNameIdx = 0
        self.datasetTypeIdx = 1
        self.classNamesIdx = 2
        self.inputFeaturesNamesIdx = 3
        self.layerNamesIdx = 4
        self.inverseOptimizationIdx = 5
        self.adversarialExamplesIdx = 6
        self.inputLayerIdx = 7
        self.groundTruthIdx = 8
        self.dataStartingIdx = 9

        self.rawValueIdx = 0
        self.saliencyMapIdx = 1
        self.filterImportanceIdx = 2
        self.filterSaliencyIdx = 3
        self.filterMaskIdx = 4
        self.filterLossImpactIdx = 5
        self.filterClusterIdx = 6

        # Define the class specific variables
        self.inputIterator = 0
        self.currentPercentileValue = 0.0

        self.setType = Dataset.TEST.value

        # Load and setup the model
        self.loadDatasets()
        self.loadModel()

        # Necessary because loadData method relies on importanceValues attribute and loadData is called in self.computeMinMaxMeanImportanceValues
        self.importanceValues = None

        # TODO: Compute the importance over number of examples
        '''
        # ensure the min max and mean importance values for every filter have been calculated for the standard model
        self.computeMinMaxMeanImportanceValues(self.standardModelName, n_examples=100)

        # load min max and mean importance values for every filter
        self.loadImportanceValues(self.standardModelName)
        '''


    def initializeTensorflow(self):
        # Create the TensorFlow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        set_session(self.sess)

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())


    def loadDatasets(self):
        if DATASET == Dataset.INTERNET_TRAFFIC:
            dataPath = "./datamark-internettrafficdata.csv"
            self.modelPath = "./modCNN_Outsidesteps1e5000lr0.001lb50batch5datamark-internettrafficdata.ctrs14000tes4000deriv1.h5"
            self.autoencoderPath = "./autoencoder-internet-traffic.h5"
            self.autoencoderWithModelPath = "./autoencoder_with_model-internet-traffic.h5"

            # name for the standard model
            self.standardModelPath = self.modelPath
            self.standardModelName = "modCNN_Outsidesteps1e5000lr0.001lb50batch5datamark-internettrafficdata.ctrs14000tes4000deriv1"

            # Load the data
            dataframe = pd.read_csv(dataPath, usecols=[1], sep=',', engine='python')
            numTrainExamples = 14000
            numTestExamples = 4000
            numValExamples = 2000

            steps = derivative = 1
            self.num_features = 1 + derivative # Include the raw signal
            self.look_back = 50
            self.num_features = 1 + self.num_features # Include the activation map for the original value

            print("Number of features: %d" % self.num_features)
            print("Number of derivatives: %d" % derivative)

            dataset = dataframe.values
            dataset = dataset.astype('float32')
            validation_split = 0.2
            train, test = dataset[0:numTrainExamples, :], dataset[(numTrainExamples):(numTrainExamples + numTestExamples), :]
            sc = StandardScaler()
            train = sc.fit_transform(train)
            test = sc.transform(test)

            train = pd.DataFrame(train)
            test = pd.DataFrame(test)

            self.trainX, self.trainY = create_dataset(train, self.look_back, steps)
            self.testX, self.testY = create_dataset(test, self.look_back, steps)

        if DATASET == Dataset.ANOMALY_DETECTION:
            dataPath = "./anomaly_dataset.pickle"
            self.modelPath = "./cnn_anomaly_dataset.h5"
            self.autoencoderPath = "./autoencoder-anomaly.h5"
            self.autoencoderWithModelPath = "./autoencoder_with_model-anomaly.h5"

            # Name for the standard model
            self.standardModelPath = self.modelPath
            self.standardModelName = "cnn_anomaly_dataset"

            # Create a custom dataset
            numTimeSteps = 50
            numChannels = 3
            numTrainExamples = 50000
            numValExamples = 5000
            numTestExamples = 10000
            plotData = False

            if not os.path.exists(dataPath):
                # np.random.seed(1)
                def sampleTimeSeries():
                    x = smoothSampling(numTimeSteps, numChannels)
                    # sc = StandardScaler()
                    # x = sc.fit_transform(x)
                    # print ("X:", x.shape)

                    y = 0  # Normal
                    if np.random.random() > 0.75:
                        # Anomaly - can be both positive or negative
                        scalingFactor = (np.sign(np.random.normal()) * np.random.normal(loc=3.0, scale=0.75))
                        anomalyIdx = np.random.randint(1, numTimeSteps - 1)

                        anomalyChannel = np.random.randint(0, numChannels)
                        y = 1.0 if anomalyChannel > 0 else 0.0  # Never introduce anomalies in the first channel

                        failedAttempts = 0
                        while x[anomalyIdx, anomalyChannel] < 1.0:  # Only take significant values
                            anomalyIdx = np.random.randint(1, numTimeSteps - 1)
                            failedAttempts += 1
                            if failedAttempts > 10:  # Reinitialize
                                x = smoothSampling(numTimeSteps, numChannels)

                        if y != 0.0:
                            x[anomalyIdx, anomalyChannel] *= scalingFactor

                    if plotData:
                        tqdm.write("Anomalous: %s" % ("True" if y else "False"))

                        fig, ax = plt.subplots()
                        # fig.set_size_inches(18.5, 10.5)
                        ax.set_title('Machine Data', color='C0')

                        # Plot the raw values
                        timeAxis = np.arange(numTimeSteps)

                        for idx, featureName in enumerate(INPUT_FEATURE_NAMES):
                            # ax.plot(timeAxis, x[:, idx], 'C' + str(idx), marker='.', markersize=10.0, label=featureName)
                            ax.plot(timeAxis, x[:, idx], 'C' + str(idx), label=featureName)
                        ax.legend()

                        ax.set_xlabel('Time-step')
                        ax.set_ylabel('Value')

                        plt.tight_layout()
                        plt.show()
                        plt.close('all')
                        # plt.savefig("./toyExample-anomaly.png", dpi=300)

                    return x, y

                self.trainX, self.trainY = [], []
                self.testX, self.testY = [], []

                for setType in ["Train", "Test"]:
                    itemList = range(numTrainExamples if setType == "Train" else numTestExamples)
                    for iterator in tqdm(itemList):
                        x, y = sampleTimeSeries()

                        if setType == "Train":
                            self.trainX.append(x)
                            self.trainY.append(y)

                        if setType == "Test":
                            self.testX.append(x)
                            self.testY.append(y)

                # Convert to numpy arrays
                self.trainX, self.trainY = np.array(self.trainX), np.array(self.trainY)
                self.testX, self.testY = np.array(self.testX), np.array(self.testY)

                # Normalize the data
                # sc = StandardScaler()
                # self.trainX = sc.fit_transform(self.trainX)
                # self.testX = sc.transform(self.testX)

                # Dump data to pickle file
                print("Saving data to file: %s" % (dataPath))
                with open(dataPath, "wb") as pickleFile:
                    pickle.dump([self.trainX, self.trainY, self.testX, self.testY], pickleFile, protocol=pickle.HIGHEST_PROTOCOL)
                print("Data saved successfully!")

            else:
                print("Loading data from file: %s" % (dataPath))
                # Load data from pickle file
                with open(dataPath, "rb") as pickleFile:
                    self.trainX, self.trainY, self.testX, self.testY = pickle.load(pickleFile)
                print("Data loaded successfully!")

        # Saves the currently loaded model name
        self.currentlyLoadedModel = self.standardModelName

        self.valX = self.trainX[numTrainExamples - numValExamples:, :, :]
        self.valY = self.trainY[numTrainExamples - numValExamples:]
        self.trainX = self.trainX[:numTrainExamples - numValExamples, :, :]
        self.trainY = self.trainY[:numTrainExamples - numValExamples]

        print("Train set | X shape: %s | Y shape: %s" % (str(self.trainX.shape), str(self.trainY.shape)))
        print("Validation set | X shape: %s | Y shape: %s" % (str(self.valX.shape), str(self.valY.shape)))
        print("Test set | X shape: %s | Y shape: %s" % (str(self.testX.shape), str(self.testY.shape)))

        if DATASET != Dataset.INTERNET_TRAFFIC:
            # Print number of anomalous sequences in the dataset
            trainAnomalies = np.sum(self.trainY == 1)
            valAnomalies = np.sum(self.valY == 1)
            testAnomalies = np.sum(self.testY == 1)
            print("Train anomalies: %d | Validation anomalies: %d | Test anomalies: %d" % (trainAnomalies, valAnomalies, testAnomalies))


    def loadCustomSingleChannelDatasets(self, dataPath):
        # Load the data
        dataframe = pd.read_csv(dataPath, usecols=[0], sep=',', engine='python')
        frameSize = len(dataframe.values)
        train_size = int(frameSize*0.7)
        test_size = frameSize - train_size

        steps = derivative = 1
        self.num_features = 1 + derivative # Include the raw signal
        self.look_back = 50
        self.num_features = 1 + self.num_features # Include the activation map for the original value

        print("Number of features: %d" % self.num_features)
        print("Number of derivatives: %d" % derivative)

        dataset = dataframe.values
        dataset = dataset.astype('float32')
        train, test = dataset[0:train_size, :], dataset[train_size:(train_size + test_size), :]
        sc = StandardScaler()
        train = sc.fit_transform(train)
        test = sc.transform(test)

        train = pd.DataFrame(train)
        test = pd.DataFrame(test)

        self.trainX, self.trainY = create_dataset(train, self.look_back, steps)
        self.testX, self.testY = create_dataset(test, self.look_back, steps)

        print("Train X shape: %s" % str(self.trainX.shape))
        print("Train Y shape: %s" % str(self.trainY.shape))
        print("Test X shape: %s" % str(self.testX.shape))
        print("Test Y shape: %s" % str(self.testY.shape))
        labels = ["Train X", "Train Y", "Test X", "Test Y"]
        shapes = [self.trainX.shape, self.trainY.shape, self.testX.shape, self.testY.shape]

        return [labels, shapes]


    def loadModel(self):
        if not os.path.exists(self.modelPath):
            assert DATASET != Dataset.INTERNET_TRAFFIC, "Error: Model generation not supported for internet traffic dataset!"

            # Train the model
            numTimeSteps = self.trainX.shape[1]
            numChannels = self.trainX.shape[2]

            regLambda = 1e-3
            actReg = 3e-3
            leakRate = 0.3

            visible = Input(shape=(numTimeSteps, numChannels))

            if BATCH_NORM:
                net = Conv1D(16, kernel_size=5, kernel_regularizer=regularizers.l2(regLambda))(visible)
                net = BatchNormalization()(net)
                net = LeakyReLU(leakRate)(net)
                net = MaxPooling1D(pool_size=2)(net)

                net = Conv1D(32, kernel_size=3, kernel_regularizer=regularizers.l2(regLambda))(net)
                net = BatchNormalization()(net)
                net = LeakyReLU(leakRate)(net)
                net = MaxPooling1D(pool_size=2)(net)

                net = Conv1D(64, kernel_size=3, kernel_regularizer=regularizers.l2(regLambda))(net)
                net = BatchNormalization()(net)
                net = LeakyReLU(leakRate)(net)

            else:
                net = Conv1D(16, kernel_size=3, kernel_regularizer=regularizers.l2(regLambda))(visible)
                net = LeakyReLU(leakRate)(net)
                if DATASET != Dataset.MAMO and DATASET != Dataset.NASA_SPACE_SHUTTLE:
                    net = MaxPooling1D(pool_size=2)(net)
                net = Conv1D(32, kernel_size=3, kernel_regularizer=regularizers.l2(regLambda))(net)
                net = LeakyReLU(leakRate)(net)
                if DATASET != Dataset.MAMO and DATASET != Dataset.NASA_SPACE_SHUTTLE:
                    net = MaxPooling1D(pool_size=2)(net)
                    net = Conv1D(64, kernel_size=3, kernel_regularizer=regularizers.l2(regLambda))(net)
                    net = LeakyReLU(leakRate)(net)

            net = Flatten()(net)

            net = Dense(len(CLASS_NAMES), kernel_regularizer=regularizers.l2(regLambda), activity_regularizer=regularizers.l1(actReg))(net)
            net = Activation('sigmoid' if len(CLASS_NAMES) == 1 else 'softmax')(net)
            model = Model(inputs=visible, outputs=net)

            optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
            model.compile(loss='binary_crossentropy' if len(CLASS_NAMES) == 1 else 'sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

            checkpoint = ModelCheckpoint(self.modelPath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', cooldown=0, min_lr=0)

            model.fit(self.trainX, self.trainY, epochs=100, validation_data=(self.valX, self.valY), shuffle=True, callbacks=[checkpoint, earlyStop, lr_reducer])

        # Load the model
        self.model = load_model(self.modelPath)
        self.model.name = "Model"
        self.currentlyLoadedModel = self.standardModelPath

        print("Model summary:")
        self.model.summary()

        out = self.model.evaluate(self.trainX, self.trainY, verbose=False)
        print("Train | Metric value: %s | Metric Name: %s" % (str(out), self.model.metrics_names))
        out = self.model.evaluate(self.valX, self.valY, verbose=False)
        print("Validation | Metric value: %s | Metric Name: %s" % (str(out), self.model.metrics_names))
        out = self.model.evaluate(self.testX, self.testY, verbose=False)
        print("Test | Metric value: %s | Metric Name: %s" % (str(out), self.model.metrics_names))

        # Get the input placeholder
        self.inputPlaceholder = self.model.input # Only single input
        outputLayers = self.model.layers
        outputLayers = [layer for layer in outputLayers if not ("flatten" in layer.name or "input" in layer.name)]  # Discard the flattening and input layers
        requiredGradients = [currentLayer.output for currentLayer in outputLayers]
        self.outputLayer = requiredGradients[-1]

        if DATASET_TYPE == Dataset.CLASSIFICATION:
            if len(CLASS_NAMES) == 1:
                self.labelsPlaceholder = tf.placeholder(tf.float32, shape=[None, 1])
            else:
                self.labelsPlaceholder = tf.placeholder(tf.int64, shape=[None, 1])
            if len(CLASS_NAMES) == 1:  # Influence will die off when sigmoid unit is saturated (multiplication with sigmoid unit in backprop)
                self.loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=self.labelsPlaceholder, output=requiredGradients[-1]))
            else:
                self.loss = tf.reduce_mean(tf.keras.backend.sparse_categorical_crossentropy(target=tf.squeeze(self.labelsPlaceholder, axis=1), output=requiredGradients[-1]))
        else:
            self.labelsPlaceholder = tf.placeholder(tf.float32, shape=[None, 1])
            self.loss = tf.reduce_mean(tf.square(self.labelsPlaceholder - requiredGradients[-1])) # Regression loss

        self.layerType = []
        self.defaultLayerNames = []

        self.layerGradientsWrtInput = []
        self.layerNames = []
        self.numLayerFilters = []
        self.slicedTensors = []
        for layerIdx, tensorToBeDifferentiated in enumerate(requiredGradients):
            layerShape = tensorToBeDifferentiated.shape
            layerName = getShortName(outputLayers[layerIdx].name, outputLayers[layerIdx].__class__.__name__)
            self.defaultLayerNames.append(layerName)
            self.layerType.append(outputLayers[layerIdx].__class__.__name__)
            print("Layer name: %s | Shape: %s | Type: %s" % (self.defaultLayerNames[-1], str(layerShape), self.layerType[-1]))
            self.numLayerFilters.append(int(layerShape[-1]))
            for filterIdx in range(layerShape[-1]):
                if len(layerShape) == 3:
                    tensorSlice = tf.expand_dims(tensorToBeDifferentiated[:, :, filterIdx], axis=-1)
                elif len(layerShape) == 2:
                    tensorSlice = tf.expand_dims(tensorToBeDifferentiated[:, filterIdx], axis=-1)
                else:
                    print("Error: Unknown tensor shape")
                    exit(-1)
                self.slicedTensors.append(tensorSlice)
                self.layerGradientsWrtInput.append(tf.gradients(tensorSlice, [self.inputPlaceholder]))
                self.layerNames.append(outputLayers[layerIdx].name + "-" + str(filterIdx))

        self.outputGradientsWrtLayer = [tf.gradients(self.outputLayer, [tensorToBeDifferentiated]) for tensorToBeDifferentiated in requiredGradients[:-1]]
        self.lossGradientsWrtLayer = [tf.gradients(self.loss, [tensorToBeDifferentiated]) for tensorToBeDifferentiated in requiredGradients[:-1]]

        # For inverse optimization and adversarial examples
        self.lossGrad = tf.gradients(self.loss, [self.inputPlaceholder])
        self.signGrad = tf.sign(self.lossGrad) # Use only the sign of the gradient
        self.outputGrad = tf.gradients(self.outputLayer, [self.inputPlaceholder])

        self.scaleGradientWrtLossValues = True
        self.adjustedFilterIndices = getAdjustedFilterIndices(self.currentlyLoadedModel)


    def prevInputButton(self):
        if self.inputIterator > 0:
            self.inputIterator -= 1
            return True
        else:
            return False


    def nextInputButton(self):
        if (self.setType == Dataset.TEST.value and (self.inputIterator < self.testX.shape[0] - 1)) or \
                (self.setType == Dataset.TRAIN.value and (self.inputIterator < self.trainX.shape[0] - 1)):
            self.inputIterator += 1
            return True
        else:
            return False


    def resetIterator(self):
        self.inputIterator = 0
        return True


    def setIterator(self, iterator):
        if (self.setType == Dataset.TEST.value and (self.inputIterator >= self.testX.shape[0] - 1)) or \
                (self.setType == Dataset.TRAIN.value and (self.inputIterator >= self.trainX.shape[0] - 1)):
            return False

        self.inputIterator = iterator
        return True


    def getIterator(self):
        return self.inputIterator


    def changePercentile(self, percentile):
        self.currentPercentileValue = percentile / 100.0


    def getModelLayerNames(self):
        if "input" not in self.layerNames[0]:
            return self.defaultLayerNames
        else:
            return self.defaultLayerNames[1:]


    def classifySequence(self, seq):
        with K.get_session().graph.as_default():
            prediction = self.model.predict(np.expand_dims(seq, axis=0))
        return prediction[0]


    def getExample(self):
        if self.setType == Dataset.TEST.value:
            X = self.testX[self.inputIterator, :, :]
            y = self.testY[self.inputIterator]
        else:
            X = self.trainX[self.inputIterator, :, :]
            y = self.trainY[self.inputIterator]
        return X, y


    def switchToTest(self):
        self.inputIterator = 0
        self.setType = Dataset.TEST.value
        return True


    def switchToTrain(self):
        self.inputIterator = 0
        self.setType = Dataset.TRAIN.value
        return True


    def getPrediction(self):
        # Get the session from Keras backend
        self.sess = K.get_session()

        X, y = self.getExample()

        # Get the prediction and the saliency for the last layer
        pred, saliency = self.sess.run([self.outputLayer, self.layerGradientsWrtInput[-1]], feed_dict={self.inputPlaceholder: np.expand_dims(X, axis=0)})
        pred = np.squeeze(pred)
        saliency = np.squeeze(saliency).T
        saliency = np.abs(saliency)  # For absolute scaling
        saliency = normalizeValues(saliency)

        return X, y, pred, saliency


    def getArchitecture(self):
        arch = []

        # Layer name, type, # filters, filter size, filter stride, output shape
        for layer in self.model.layers:
            # layerDetails = {}
            # layerDetails['name'] = layer.name
            # layerDetails['type'] = getLayerType(layer.name)
            #
            # weightMatrix = layer.get_weights()
            # if len(weightMatrix) > 0:
            #     layerDetails['num_filters'] = int(weightMatrix[0].shape[-1])
            #     layerDetails['filter_size'] = int(weightMatrix[0].shape[0])
            # layerDetails['filter_stride'] = -1
            # layerDetails['output_shape'] = [int(dim) for dim in layer.output_shape[1:]]
            # arch.append(layerDetails)
            arch.append(layer.get_config())

        return arch


    def loadData(self, fastMode=Modes.FULL.value, visualizationFilters=False, verbose=False):
        # Get the session from Keras backend
        self.sess = K.get_session()

        serviceOutput = []
        if self.setType == Dataset.TEST.value:
            currentInput = self.testX[self.inputIterator, :, :]
            currentLabel = self.testY[self.inputIterator]
        else:
            currentInput = self.trainX[self.inputIterator, :, :]
            currentLabel = self.trainY[self.inputIterator]

        # Append dataset type
        serviceOutput.append(str(DATASET.name))
        serviceOutput.append(str(DATASET_TYPE.name))
        serviceOutput.append(CLASS_NAMES)
        serviceOutput.append(INPUT_FEATURE_NAMES)

        # layerNames[0] contains "input" if a pruned model has been loaded
        serviceOutput.append(self.getModelLayerNames())

        if fastMode == Modes.FULL.value:
            # Add inverse optimization and adversarial examples output here
            startingSeries, startingSeriesForecast, startSerSaliencyMap, invOptimizedSeries, invOptimizedForecast, invOptSaliencyMap, advExampleOrig, forecastValueAdvOrig, advExSaliencyMap = self.performInverseOptimizationAndAdvAttack()
            serviceOutput.append([startingSeries.tolist(), startingSeriesForecast.tolist(), startSerSaliencyMap.tolist(),
                                  invOptimizedSeries.tolist(), invOptimizedForecast.tolist(), invOptSaliencyMap.tolist()]) # Inverse optimization
            serviceOutput.append([advExampleOrig.tolist(), forecastValueAdvOrig.tolist(), advExSaliencyMap.tolist()]) # Adversarial examples

        else:
            # Add the inverse optimization and adversarial examples output
            serviceOutput.append("Not computed")
            serviceOutput.append("Not computed")

        # Add the raw input and the label
        serviceOutput.append(currentInput.T.tolist())
        serviceOutput.append(currentLabel.T.tolist())

        # Iterate over the layers
        layerIdx = 0
        plotIdxRow = 0
        plotIdxCol = 0
        prevLayerName = None
        for idx, inputGradients in enumerate(self.layerGradientsWrtInput):
            currentLayerRootName = self.layerNames[idx]
            currentLayerRootName = currentLayerRootName[:currentLayerRootName.rfind('-')]

            computeGradientWrtInputOnly = True

            # If new layer encountered
            if (prevLayerName != currentLayerRootName):
                serviceOutput.append([])

                # # Add the min and max bounds for the plot (vertical)
                # self.plotBounds.append((self.currentPosition[2], self.currentPosition[2] + self.plotHeight))

                if (layerIdx < len(self.outputGradientsWrtLayer)):
                    gradientsWrtInput, gradientsWrtOutput, gradientsWrtLoss, tensorSlice, loss = self.sess.run([self.layerGradientsWrtInput[idx], \
                            self.outputGradientsWrtLayer[layerIdx], self.lossGradientsWrtLayer[layerIdx], self.slicedTensors[idx], self.loss], \
                            feed_dict={self.inputPlaceholder: np.expand_dims(currentInput, axis=0), self.labelsPlaceholder: np.array(currentLabel).reshape(1, 1)})

                    def removeBatchDim(input):
                        # Since we are computing gradient wrt to only tensor, the returned list should be of length 1
                        assert(len(input) == 1)
                        input = input[0]  # tf.Gradients returns a list
                        assert(input.shape[0] == 1)
                        return input[0]  # Batch size is always 1

                    gradientsWrtOutput = removeBatchDim(gradientsWrtOutput)
                    gradientsWrtInput = removeBatchDim(gradientsWrtInput)
                    gradientsWrtLoss = removeBatchDim(gradientsWrtLoss)

                    if verbose:
                        print("Input grads shape:", gradientsWrtInput.shape)
                        print("Output grads shape:", gradientsWrtOutput.shape)

                    def cummulateGradients(input):
                        # Combine the gradient for each filter to determine the impact of each filter on the final outcome
                        input = np.abs(input)  # Absolute values should have been taken
                        if len(input.shape) > 1:  # No reduction for dense layers
                            input = np.sum(input, axis=0)
                        else:
                            input = input.copy()
                        return input

                    perFilterGradientWrtOutput = cummulateGradients(gradientsWrtOutput)
                    gradientsWrtOutput = normalizeValues(gradientsWrtOutput)

                    perFilterGradientWrtLoss = cummulateGradients(gradientsWrtLoss)

                    # Scale values
                    perFilterGradientWrtOutput = np.abs(perFilterGradientWrtOutput)  # For absolute scaling
                    perFilterGradientWrtOutput = normalizeValues(perFilterGradientWrtOutput)

                    if self.scaleGradientWrtLossValues:
                        perFilterGradientWrtLoss = np.abs(perFilterGradientWrtLoss)

                        # Scale values (gradient w.r.t. loss)
                        perFilterGradientWrtLoss = normalizeValues(perFilterGradientWrtLoss)

                    if verbose:
                        print("Filter gradient shape:", perFilterGradientWrtOutput.shape, "| Filter gradient value:", perFilterGradientWrtOutput)
                        print("Loss gradient shape:", perFilterGradientWrtLoss.shape, "| Loss gradient value:", perFilterGradientWrtLoss)

                    # Create the filter mask with percentile score
                    percentileIndex = int(np.round(perFilterGradientWrtOutput.shape[0] * self.currentPercentileValue))
                    sortedIndices = np.argsort(perFilterGradientWrtOutput)
                    percentileMask = np.zeros_like(perFilterGradientWrtOutput, dtype=np.bool)
                    percentileMask[sortedIndices[percentileIndex:]] = True

                    computeGradientWrtInputOnly = False

                else:
                    gradientsWrtInput = None
                    perFilterGradientWrtOutput = None
                    percentileMask = None

            if computeGradientWrtInputOnly:
                gradientsWrtInput, tensorSlice, loss, output = self.sess.run([self.layerGradientsWrtInput[idx], self.slicedTensors[idx], self.loss, self.outputLayer], \
                                feed_dict={self.inputPlaceholder: np.expand_dims(currentInput, axis=0), self.labelsPlaceholder: np.array(currentLabel).reshape(1, 1)})

            # Create a new subplot for a new layer
            if (prevLayerName != currentLayerRootName):
                numSubPlotElements = math.ceil(math.sqrt(self.numLayerFilters[layerIdx]))
                plotIdxRow = 0
                plotIdxCol = 0
                layerIdx += 1

            else:
                plotIdxCol += 1
                if plotIdxCol == numSubPlotElements:
                    plotIdxRow += 1
                    plotIdxCol = 0

            prevLayerName = currentLayerRootName

            # Scale values
            gradientsWrtInput = np.abs(gradientsWrtInput)  # For absolute scaling
            saliencyMap = normalizeValues(gradientsWrtInput)
            saliencyMap = np.squeeze(saliencyMap)

            currentPlotIdx = (plotIdxRow * numSubPlotElements) + plotIdxCol

            # Transpose the values since the default size is [Rows, Cols]
            if len(gradientsWrtOutput.shape) > 1:
                filterSaliency = gradientsWrtOutput[:, currentPlotIdx].flatten()
            else:
                filterSaliency = gradientsWrtOutput[currentPlotIdx]

            filterImportance = perFilterGradientWrtOutput[currentPlotIdx] if perFilterGradientWrtOutput is not None else 1.0
            filterLossImpact = perFilterGradientWrtLoss[currentPlotIdx]
            filterMask = percentileMask[currentPlotIdx] if percentileMask is not None else True

            if verbose:
                layerName = self.layerNames[idx]
                print("Computing output for layer:", layerName)
                print("Saliency map shape:", saliencyMap.shape)

            serviceOutput[-1].append([])
            serviceOutput[-1][-1].append(tensorSlice.flatten().tolist())
            serviceOutput[-1][-1].append(saliencyMap.T.tolist())
            serviceOutput[-1][-1].append(float(filterImportance))
            serviceOutput[-1][-1].append(filterSaliency.tolist()) # New addition (adjust the corresponding idx)
            serviceOutput[-1][-1].append(bool(filterMask))
            serviceOutput[-1][-1].append(float(filterLossImpact))

        # Remove redundant input-layer if pruned model is currently loaded
        if "input" in self.layerNames[0]:
            del serviceOutput[self.dataStartingIdx]

        if fastMode != Modes.MINIMAL.value:
            # Iterate over all the filters to compute the rank of filter clusters
            computeFilterClusters(self, serviceOutput, visualizeFilters=visualizationFilters, verbose=verbose)

        if fastMode == Modes.FULL.value:
            print("Loss:", loss, "| Y:", np.squeeze(currentLabel), "| Output:", np.squeeze(output))

        return serviceOutput


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~ Fliter Clustering ~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    def performFilterClustering(self):
        serviceOutput = self.loadData()
        computeFilterClusters(self, serviceOutput, visualizeFilters=False)


    def visualizeFilterClusters(self):
        serviceOutput = self.loadData(visualizationFilters=True)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~ Network Pruning ~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    def computePruningFilterSet(self, dataMode, importanceMode, numberOfFilter, layerSelection, importanceSelection, dataset):
        # datamode 0 over current example 1 for over all examples
        self.setType = dataset  # Select the set based on the defined param
        serviceOutput = self.loadData()
        convIds = [i + self.dataStartingIdx for i in range(len(serviceOutput[self.layerNamesIdx])) if "conv" in serviceOutput[self.layerNamesIdx][i]]
        importanceValues = []

        if importanceSelection == Modes.IMPORTANCE.value:
            valueIdx = self.filterImportanceIdx
        else:
            valueIdx = self.filterLossImpactIdx

        if dataMode == Modes.SELECTED_EXAMPLES.value and importanceMode < Modes.CLUSTER_REPRESENTATIVES.value:
            for layerIdx in convIds:
                layerList = []
                for filterO in serviceOutput[layerIdx]:
                    layerList.append(filterO[valueIdx])
                importanceValues.append(layerList)
            result = computeIndices(importanceValues, importanceMode, numberOfFilter, layerSelection, len(convIds))

        if dataMode == Modes.ALL_EXAMPLES.value and importanceMode < Modes.CLUSTER_REPRESENTATIVES.value:
            backupInputIterator = self.inputIterator
            self.inputIterator = 0
            serviceOutput = self.loadData(fastMode=Modes.MINIMAL.value)
            # Initial design
            for layerIdx in convIds:
                layerList = []
                for filterO in serviceOutput[layerIdx]:
                    layerList.append(filterO[valueIdx])
                importanceValues.append(layerList)
            self.inputIterator += 1
            while self.inputIterator < self.trainX.shape[0] - 1:
                serviceOutput = self.loadData(fastMode=Modes.MINIMAL.value)
                for layerIdx, serviceIdx in enumerate(convIds):
                    for filterIdx, filterO in enumerate(serviceOutput[serviceIdx]):
                        importanceValues[layerIdx][filterIdx] = filterO[valueIdx]
                self.inputIterator += 1
            self.inputIterator = backupInputIterator
            serviceOutput = self.loadData()
            result = computeIndices(importanceValues, importanceMode, numberOfFilter, layerSelection, len(convIds))

        if dataMode == Modes.SELECTED_EXAMPLES.value and importanceMode == Modes.CLUSTER_REPRESENTATIVES.value:
            result = computeRepresentatives(self, serviceOutput, layerSelection, convIds)

        if dataMode == Modes.ALL_EXAMPLES.value and importanceMode == Modes.CLUSTER_REPRESENTATIVES.value:
            result = []
            backupInputIterator = self.inputIterator
            self.inputIterator = 0
            while self.inputIterator < self.trainX.shape[0] - 1:
                serviceOutput = self.loadData(fastMode=Modes.PARTIAL.value)
                result.append(computeRepresentatives(serviceOutput, layerSelection, convIds))
                self.inputIterator += 1
            result = matchRepresentatives(result)
            self.inputIterator = backupInputIterator
            serviceOutput = self.loadData()

        return result


    def computePruningFilterSetFromFile(self, mode, submode, percentile, reverse, examples, importance, dataset):
        modelName = self.currentlyLoadedModel
        dirpath = os.path.join(".", "Statistics")
        filepath = os.path.join(dirpath, modelName)
        ensureDirExists(filepath)

        # If number of examples is zero or negative, compute the statistics over the entire dataset
        if examples < 1:
            examples = self.testX.shape[0] if dataset == Dataset.TEST.value else self.trainX.shape[0]

        if reverse == 1:
            percentile = 100 - percentile

        if importance == Modes.IMPORTANCE.value:
            importanceIdx = visualizationToolbox.filterImportanceIdx
            measure = "importance"
        else:
            importanceIdx = visualizationToolbox.filterLossImpactIdx
            measure = "loss"

        if mode == Modes.COMPUTE_RANDOM.value:
            filterObject = randomUseless(visualizationToolbox, percentile)

        if mode == Modes.COMPUTE_PERCENTILE.value:
            filepath = os.path.join(filepath, "MinMaxMean_" + measure + ".npy")
            if not os.path.isfile(filepath):
                computeAndSaveImportanceValues(self, filepath, examples, self.filterImportanceIdx, dataset)
            removementSet = computeRemovementSets(self, filepath, percentile)
            filterObject = removementSet[submode][0]

        if mode == Modes.COMPUTE_REPRESENTATIVE.value:
            filepath = os.path.join(filepath, "MeanRepresentatives_" + measure + ".npy")
            if not os.path.isfile(filepath):
                computeRepresentativeValues(self, filepath, examples, self.filterImportanceIdx, dataset)
            mostRepresentatives = computeMostTimesRepresentives(self, filepath, dataset)
            filterObject = keepOnlyRepresentants(self, mostRepresentatives, dataset)

        if reverse == 1:
            filterObject = reverseFilterList(visualizationToolbox, filterObject)

        return filterObject


    def performNetworkPruning(self, epochs, indices, name, mode):
        dirpath = "."
        dirpathAdjustedFilters = os.path.join(dirpath, 'adjustedFilters')

        # File that stores the indexes of filters of the currently loaded model that have been adjusted (weights and biases set to 0)
        filepathOldModel = os.path.join(dirpathAdjustedFilters, self.currentlyLoadedModel)

        # File that will store the indexes of filters of the new model that have been adjusted (weights and biases set to 0)
        filepathNewModel = os.path.join(dirpathAdjustedFilters, name)

        dirpathModelNames = os.path.join(dirpath, 'prunedModels')
        filepathModelNames = os.path.join(dirpathModelNames, 'prunedModelNames')

        # Check if a model with the same name exists already
        try:
            with open(filepathModelNames, 'r') as file:
                override = (name + os.linesep) in file
        except OSError:
            override = False

        if override:
            pass
        else:
            # Only append name to file if there doesn't exist a model with the same name already.
            try:
                with open(filepathModelNames, 'a+') as file:
                    file.write(name + '\n')
            except OSError:
                print('File at ' + filepathModelNames + ' could not be opened or created')
                return jsonify({'status': 'error', 'msg': 'File at ' + filepathModelNames + ' could not be opened or created'})

        # read the adjustedFilters file of the old model and convert the index string to another format
        indiceString = []
        try:
            with open(filepathOldModel, "r") as file:
                indiceString = file.read() # Get indexes of the adjusted filters of the currently loaded model
                adjustedFilterList = indiceStringToList(indiceString) # convert string to python list
        except OSError:
            print("Error: Unable to open the adjusted filter files at " + filepathOldModel)
            adjustedFilterList = []

        if mode == Modes.PRUNE.value:
            # Remove filters specified in the string indices
            prunedModel = pruneNetwork(K.get_session(), self.model, indices)

            # Train the new model for a few epochs
            score, scoreP = finetunePrunedModel(K.get_session(), prunedModel, self, epochs)

            # Update the adjusted filters file (remove adjusted indexes if filter at index has been pruned, decrement every filter index greater than a pruned filter (to keep the indexes correct after pruning))
            for index1, prunedFilterLayerNew in enumerate(indices):  # Indices holds the indexes of the filters to be pruned
                if index1 >= len(adjustedFilterList):
                    break

                # Remove pruned filters from adjusted filter list
                adjustedFilterList[index1] = [number for number in adjustedFilterList[index1] if not number in prunedFilterLayerNew]

                # Decrement every adjusted index if it's greater than a pruned index. (to keep the indexes correct after pruning(removing) filters from the model)
                for index2 in range(len(prunedFilterLayerNew)):
                    prunedFilterIndex = prunedFilterLayerNew[index2]
                    adjustedFilterList[index1] = decrementIfGreaterThan(intList=adjustedFilterList[index1], threshold=prunedFilterIndex)
                    prunedFilterLayerNew = decrementIfGreaterThan(intList=prunedFilterLayerNew, threshold=prunedFilterIndex) # indexes that are pruned, also need to be adjusted after "pruning" a filter

            # Convert back to string format to save them again
            indiceString = listToIndiceString(adjustedFilterList)

        elif mode == Modes.ADJUST.value:
            # Set filter weights and biases to 0. For filters specified in indices
            prunedModel = adjustWeights(K.get_session(), self.model, indices)

            # Evaluate the new model
            score, scoreP = finetunePrunedModel(K.get_session(), prunedModel, self, -1)

            # Update the adjusted filters file (add new adjusted indexes to new file for new model)
            combined = joinInnerLists(indices, adjustedFilterList) # combine the adjusted filter indexes from the old model and the newly adjusted filter indexes

            # Sort them
            for sublist in combined:
                sublist.sort()

            # Convert back to string format to save them again
            indiceString = listToIndiceString(combined)

        else:
            print("Error: Only two modes supported!")
            return

        if name != "":  # Don't save the model for 'Faithfulness' evaluation
            # Save updated adjusted indices in the file for the new model
            with open(filepathNewModel, "w") as file:
                file.write(indiceString)

            save_model(model=prunedModel, filepath=os.path.join("prunedModels", name + ".h5"), overwrite=True, include_optimizer=True)

        return [score, scoreP]


    def performTesting(self):
        print("Evaluation results:")
        with K.get_session().graph.as_default():
            score = self.model.evaluate(x=self.testX, y=np.reshape(self.testY, self.testY.shape[0]), verbose=1)

        for idx, metricName in enumerate(self.model.metrics_names):
            print("Metric:", metricName, "| Test value:", score[idx])

        return [self.model.metrics_names, score]


    def loadImportanceValues(self, modelName):
        # Reads an importance statistics file that stores min, max and mean importance values of every filter of a model
        output = []
        path = os.path.join(".", os.path.join("ImportanceStatistics", modelName))
        try:
            txt_file = open(path, "r")
            last_name = ""
            for line in txt_file:
                name, min_val, max_val, mean_val = parseLine(line)

                if name != last_name:
                    output.append([])
                    last_name = name

                output[-1].append((min_val, max_val, mean_val))

            self.importanceValues = output
            print("Min, max, mean importance values loaded for model: " + modelName)
        except OSError:
            self.importanceValues = None


    def computeMinMaxMeanImportanceValues(self, modelName, n_examples):
        # computes min, max, mean importance values for every filter of the currently loaded model
        dirpath = os.path.join(".", "ImportanceStatistics")
        filepath = os.path.join(dirpath, modelName)
        ensureDirExists(dirpath)
        if not os.path.isfile(filepath):
            temporary = self.importanceValues
            self.importanceValues = None
            computeAndSaveImportanceValues(self, filepath, n_examples, self.filterImportanceIdx)
            self.importanceValues = temporary


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~ Inverse Optimization ~~~~~~~~~~~~~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    def performInverseOptimization(self, y, iterations=1000, stepSize=1.0, randomStart=True, optimizeOnlyRawValues=True,
                                   smoothSampling=False, addSmoothnessPrior=False):
        inputPlaceholder = self.inputPlaceholder
        gtPlaceholder = self.labelsPlaceholder

        if randomStart:
            optimizedInput = np.random.normal(loc=0.0, scale=0.25, size=(1, self.trainX.shape[1], self.trainX.shape[2])) # Loc = Mean, Scale = Std. Dev.
            if smoothSampling:
                optimizedInput[0, :, 0] = smoothConsistentSampling(self.trainX.shape[1])
            elif addSmoothnessPrior:
                optimizedInput[0, :, 0] = smooth(optimizedInput[0, :, 0], 10)
        else:
            optimizedInput = np.zeros((1, self.trainX.shape[1], self.trainX.shape[2]))

        # Zero out other channels
        # if optimizeOnlyRawValues:
        #     for i in range(1, optimizedInput.shape[2]):
        #         optimizedInput[0, :, i] = np.zeros((self.trainX.shape[1]))

        # Get input saliency and forecast for the starting series
        startingSeries = optimizedInput.copy()
        startSerSaliencyMap = self.sess.run(self.outputGrad, feed_dict={self.inputPlaceholder: startingSeries})[0]
        startSerSaliencyMap = normalizeValues(np.abs(startSerSaliencyMap))
        startSerForecast = self.sess.run(self.model.layers[-1].output, feed_dict={inputPlaceholder: startingSeries})[0]

        for i in range(iterations):
            grad = self.sess.run(self.lossGrad, feed_dict={inputPlaceholder: optimizedInput, gtPlaceholder: y.reshape(1, 1)})
            grad = grad[0]

            # Subtract the gradient in order to decrease the loss
            if optimizeOnlyRawValues:
                optimizedInput[0, :, 0] -= (stepSize * grad[0, :, 0])
            else:
                optimizedInput -= (stepSize * grad)

        # Get input saliency and forecast for the optimized input
        saliencyMap = self.sess.run(self.outputGrad, feed_dict={self.inputPlaceholder: optimizedInput})[0]
        saliencyMap = normalizeValues(np.abs(saliencyMap))
        forecastValue = self.sess.run(self.model.layers[-1].output, feed_dict={inputPlaceholder: optimizedInput})[0]

        return startingSeries, startSerSaliencyMap, startSerForecast, optimizedInput, forecastValue, saliencyMap


    def performFGSMAttack(self, x, y, performIterativeAttack=False, useFGSM=True, iterations=100, stepSize=0.5,
                          alpha=1e-3, optimizeOnlyRawValues=True, numValuesToOptimize=None):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performFGSMTwoAttack(self, x, y, performIterativeAttack=True, useFGSM=True, iterations=100, stepSize=0.5,
                             alpha=1e-3, optimizeOnlyRawValues=True, numValuesToOptimize=None):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performIterativeFGSMAttack(self, x, y, performIterativeAttack=True, useFGSM=True, iterations=100, stepSize=0.5,
                                   alpha=1e-3, optimizeOnlyRawValues=True, numValuesToOptimize=None):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performGMAttack(self, x, y, performIterativeAttack=False, useFGSM=False, iterations=100, stepSize=10.0,
                        alpha=1e-1, optimizeOnlyRawValues=True, numValuesToOptimize=1):
        return self.performAttack(x, y, performIterativeAttack, useFGSM, iterations, stepSize, alpha,
                                  optimizeOnlyRawValues, numValuesToOptimize)


    def performAttack(self, x, y, performIterativeAttack=False, useFGSM=False, iterations=100, stepSize=0.5, alpha=1e-3,
                      optimizeOnlyRawValues=True, numValuesToOptimize=None):
        inputPlaceholder = self.inputPlaceholder
        gtPlaceholder = self.labelsPlaceholder

        originalInput = np.copy(x)
        optimizedInput = x
        if optimizeOnlyRawValues:
            optimizedInput[0, :, 1] = np.zeros((self.trainX.shape[1]))

        for i in range(iterations):
            # grad = self.sess.run(gradient, feed_dict={inputPlaceholder: optimizedInput, gtPlaceholder: y})[0]
            [currentLoss, grad] = self.sess.run([self.loss, self.signGrad if useFGSM else self.lossGrad],
                                                feed_dict={inputPlaceholder: optimizedInput, gtPlaceholder: y.reshape(1, 1)})
            grad = grad[0]
            # print("Loss:", currentLoss) # Attack log

            # Clip the grad
            if numValuesToOptimize is not None:
                for channel in range(grad.shape[2]):
                    sortedIdx = np.argsort(grad[0, :, channel])[::-1]  # Sort descending
                    # topIdx = sortedIdx[:numValuesToOptimize]
                    bottomIdx = sortedIdx[numValuesToOptimize:]
                    grad[0, bottomIdx, channel] = 0.0

            if not performIterativeAttack:
                # Add the gradient in order to increase the loss
                if optimizeOnlyRawValues:
                    optimizedInput[0, :, 0] += (stepSize * grad[0, :, 0])
                else:
                    optimizedInput += (stepSize * grad)

                break

            # Add the gradient in order to increase the loss
            if optimizeOnlyRawValues:
                optimizedInput[0, :, 0] = np.clip(optimizedInput[0, :, 0] + (alpha * grad[0, :, 0]),
                                                  originalInput[0, :, 0] - stepSize, originalInput[0, :, 0] + stepSize)
            else:
                optimizedInput = np.clip(optimizedInput + (alpha * grad), originalInput - stepSize,
                                         originalInput + stepSize)

        forecastValue = self.sess.run(self.model.layers[-1].output, feed_dict={inputPlaceholder: optimizedInput})[0]

        # Get input saliency (assuming only one output)
        saliencyMap = self.sess.run(self.outputGrad, feed_dict={self.inputPlaceholder: optimizedInput})[0]
        saliencyMap = normalizeValues(np.abs(saliencyMap))

        return optimizedInput, forecastValue, saliencyMap


    def performInverseOptimizationAndAdvAttack(self):
        gtPoint = self.testY[self.inputIterator]
        startingSeries, startSerSaliencyMap, startingSeriesForecast, invOptimizedSeries, invOptimizedForecast, invOptSaliencyMap = \
            self.performInverseOptimization(gtPoint, stepSize=1e-2, smoothSampling=True, iterations=1000, optimizeOnlyRawValues=False)

        advExampleOrig, forecastValueAdvOrig, advExSaliencyMap = self.performIterativeFGSMAttack(
            np.copy(self.testX[self.inputIterator][np.newaxis, :, :]), gtPoint, alpha=1e-4, stepSize=1e-1,
            iterations=1000, optimizeOnlyRawValues=False)

        return startingSeries, startingSeriesForecast, startSerSaliencyMap, invOptimizedSeries, invOptimizedForecast, invOptSaliencyMap, advExampleOrig, forecastValueAdvOrig, advExSaliencyMap
