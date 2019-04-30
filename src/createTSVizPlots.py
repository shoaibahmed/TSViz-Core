#!/bin/python

# Import required packages
import os
import colorsys

import numpy as np
import cv2
import shutil

# RESTful data loading
import requests
import json

FILTER_WIDTH = 1800 * 2
FILTER_HEIGHT = 1200 * 2
RESIZE_RATIO = 0.25

# Define the API data indices
datasetNameIdx = 0
datasetTypeIdx = 1
classNamesIdx = 2
inputFeaturesNamesIdx = 3
layerNamesIdx = 4
inverseOptimizationIdx = 5
adversarialExamplesIdx = 6
inputLayerIdx = 7
groundTruthIdx = 8
dataStartingIdx = 9

rawValueIdx = 0
saliencyMapIdx = 1
filterImportanceIdx = 2
filterSaliencyIdx = 3
filterMaskIdx = 4
filterLossImpactIdx = 5
filterClusterIdx = 6


# @numba.jit(nopython=True, cache=True, parallel=True)
def createPlotImg(filterName, filterImportance, filterCluster, maxFilterCluster, rawValues, saliency, plotForecast=False,
                  filterHeight=FILTER_HEIGHT, filterWidth=FILTER_WIDTH, showFilterImportance=False, showSaliency=False,
                  showFilterCluster=False, plotGridLines=True, matPlotLibStyle=False,
                  color=None):

    # Security checks
    assert len(rawValues.shape) == 1 or (matPlotLibStyle and len(
        rawValues.shape) == 2), "Error: MatPlotLib style only takes multi-channel input"
    assert ((not matPlotLibStyle) or (matPlotLibStyle and (color is not None) and (rawValues.shape[0] == len(color)))), \
        "Error: Number of defined color and the number of channels does not match"

    # Disable all other features in MatPlotLib style
    if matPlotLibStyle:
        plotGridLines = False
        plotAxisLabels = True
        showFilterCluster = False
        showSaliency = False
        showFilterImportance = False
    else:
        plotAxisLabels = plotGridLines

    # Create the plot
    dataPlot = np.ones((filterHeight, filterWidth, 4), dtype=np.float32)

    # Add gradient based background to the plot
    if showFilterCluster:
        # Map the cluster number to color
        colorsHSV = [(x * (1.0 / (maxFilterCluster + 1)), 0.5, 0.5) for x in range(maxFilterCluster + 1)]
        colorsRGB = list(map(lambda x: colorsys.hsv_to_rgb(*x), colorsHSV))
        # print("HSV:", colorsHSV, "| RGB:", colorsRGB)

        # All the color to the cluster and the rect around it
        currentClusterColor = colorsRGB[filterCluster]
        currentClusterColor = np.array([currentClusterColor[0], currentClusterColor[1], currentClusterColor[2], 1.0])

        startColor = currentClusterColor[:3].copy()
        endColor = np.array([0.95, 0.95, 0.95])
        startColor = (startColor + endColor) / 2.0  # Update the start color

    else:
        startColor = np.array([0.75, 0.75, 0.75])
        endColor = np.array([0.95, 0.95, 0.95])

    colorStride = (endColor - startColor) / filterWidth
    currentColor = startColor
    for i in range(filterWidth):
        dataPlot[:, i, :] = np.concatenate((currentColor, [0.9]))
        currentColor += colorStride

    # Create the layout of the table
    titleHeight = int(dataPlot.shape[0] / 20)

    skipStride = 2e-1
    for i in range(titleHeight):
        skip = int(np.ceil(i ** (1.4 if filterHeight <= 1000 else 1.3 + skipStride)))
        for j in range(dataPlot.shape[1] - skip):
            if showFilterImportance:
                dataPlot[i, j, :] = [1.0, 0.0, 0.0, filterImportance]  # RGB
            else:
                dataPlot[i, j, :] = [0.0, 0.0, 0.0, 1.0]

    # Add the plot title
    plotDim = min(filterHeight, filterWidth)
    cv2.putText(dataPlot, filterName,
                (int(dataPlot.shape[1] / 2) - (len(filterName) * int(plotDim / 67)), 45 * int(plotDim / 1000)),
                cv2.FONT_HERSHEY_SIMPLEX, 1 * int(plotDim / 1000), (1.0, 1.0, 1.0, 1.0), 8, cv2.LINE_AA)

    # Normalize the points
    def normalizeValues(values, returnScale=False):
        minVal = np.min(values)
        maxVal = np.max(values)
        if (maxVal - minVal) == 0:
            values = (values - minVal)
        else:
            values = (values - minVal) / (maxVal - minVal)
        if returnScale:
            return values, minVal, maxVal
        else:
            return values

    rawValues, minVal, maxVal = normalizeValues(rawValues, returnScale=True)

    if matPlotLibStyle:
        allValues = rawValues.copy()
        rawValues = allValues[0, :]

    # Add the data points
    padding = int(filterWidth / 15)  # 50 on each side
    plotWidth = filterWidth - (2 * padding)
    plotHeight = filterHeight - titleHeight - (2 * padding)
    pointStride = plotWidth / (rawValues.shape[0] - 1)
    borderRelaxation = int(padding / 2)

    if plotGridLines:
        # Add grid lines to the plot
        gridLineApproxStride = 300
        numVerticalGridLines = int(plotWidth / gridLineApproxStride) # Discard the fraction part
        numHorizontalGridLines = int(plotHeight / gridLineApproxStride)
        gridLineVerticalStride = int(plotWidth / numVerticalGridLines) # Recompute to compensate for the fractional part
        gridLineHorizontalStride = int(plotHeight / numHorizontalGridLines)

        gridLineColor = (0.5, 0.5, 0.5, 0.5)
        girdLineWidth = 5

        # Vertical grid lines
        currentPoint = padding
        while currentPoint < (plotWidth + padding + borderRelaxation):
            cv2.line(dataPlot, (currentPoint, titleHeight + padding - borderRelaxation), (currentPoint, titleHeight + padding + plotHeight + borderRelaxation), gridLineColor, girdLineWidth)
            currentPoint += gridLineVerticalStride

        # Horizontal grid lines
        currentPoint = padding + titleHeight
        while currentPoint < (plotHeight + titleHeight + padding + borderRelaxation):
            cv2.line(dataPlot, (padding - borderRelaxation, currentPoint), (padding + plotWidth + borderRelaxation, currentPoint), gridLineColor, girdLineWidth)
            currentPoint += gridLineHorizontalStride

    if plotAxisLabels:
        # Add the axis labels
        text = "Timesteps"
        cv2.putText(dataPlot, text, (int(dataPlot.shape[1] / 2) - (len(text) * int(plotDim / 200)),
                    filterHeight - 30 * int(plotDim / 1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * int(plotDim / 1000),
                    (0.0, 0.0, 0.0, 1.0), 4, cv2.LINE_AA)

        # Copying to make sure the texture is preserved
        textWidth = 250
        textHeight = 40
        startingWidth = 50 # Could be replaced with border relaxation
        startingHeight = int(dataPlot.shape[0] / 2) - (len(text) * int(plotDim / 200))
        textRegion = dataPlot[startingHeight:startingHeight + textWidth, startingWidth:startingWidth + textHeight, :].copy()
        textRegion = np.transpose(textRegion, axes=(1, 0, 2)) # Get text into horizontal form
        textRegion = np.ascontiguousarray(textRegion) # Problem due to transpose
        text = "Raw value"
        cv2.putText(textRegion, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * int(plotDim / 1000), (0.0, 0.0, 0.0, 1.0), 4, cv2.LINE_AA)

        textRegion = np.transpose(textRegion, axes=(1, 0, 2)) # Get text into vertical form
        dataPlot[startingHeight:startingHeight + textWidth, startingWidth:startingWidth + textHeight, :] = textRegion[::-1, :, :] # Since gradient is vertical, inversion doesn't matter

        # Add scale values
        # Horizontal Scale
        text = "1"
        cv2.putText(dataPlot, text, (padding - (len(text) * int(plotDim / 400)),
                    filterHeight - 30 * int(plotDim / 1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * int(plotDim / 1000),
                    (0.0, 0.0, 0.0, 1.0), 2, cv2.LINE_AA)

        text = str(rawValues.shape[0])
        cv2.putText(dataPlot, text, (filterWidth - padding - (len(text) * int(plotDim / 400)),
                    filterHeight - 30 * int(plotDim / 1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * int(plotDim / 1000),
                    (0.0, 0.0, 0.0, 1.0), 2, cv2.LINE_AA)

        # Vertical Scale
        text = "{:.2f}".format(minVal)
        cv2.putText(dataPlot, text, (startingWidth - (len(text) * int(plotDim / 400)),
                    titleHeight + padding + plotHeight + int(plotDim / 400)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * int(plotDim / 1000), (0.0, 0.0, 0.0, 1.0), 2, cv2.LINE_AA)

        text = "{:.2f}".format(maxVal)
        cv2.putText(dataPlot, text, (startingWidth - (len(text) * int(plotDim / 400)),
                    titleHeight + padding + int(plotDim / 400)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * int(plotDim / 1000), (0.0, 0.0, 0.0, 1.0), 2, cv2.LINE_AA)

    # Main plot border
    borderWidth = 20
    borderStride = int(borderWidth / 2)
    cv2.rectangle(dataPlot, (borderStride, borderStride), (filterWidth - borderStride, filterHeight - borderStride),
                  (0.0, 0.0, 0.0, 1.0), borderWidth)

    # Draw the data containing rectangle
    cv2.rectangle(dataPlot, (padding - borderRelaxation, titleHeight + padding - borderRelaxation),
                  (padding + plotWidth + borderRelaxation, titleHeight + padding + plotHeight + borderRelaxation),
                  (0.0, 0.0, 0.0, 1.0), 10)

    '''
    f = interpolate.interp1d(np.arange(rawValues.shape[0]), rawValues, kind='cubic')
    rawValuesInterpolated = f(rawValues)
  
    print ("Raw values shape:", rawValues.shape)
    print ("Interpolated raw values shape:", rawValuesInterpolated.shape)
    '''

    def drawPoint(i, channelIdx=None):
        # Lowest value should be at bottom
        # currentPoint = (int((i * pointStride) + padding), int((rawValues[i] * plotHeight) + titleHeight + padding))
        currentPoint = (int((i * pointStride) + padding), int((1.0 - rawValues[i]) * plotHeight) + titleHeight + padding)

        # Add lines in between
        if i < rawValues.shape[0] - 1:
            # Lowest value should be at bottom
            # nextPoint = (int(((i + 1) * pointStride) + padding), int((rawValues[i + 1] * plotHeight) + titleHeight + padding))
            nextPoint = (int(((i + 1) * pointStride) + padding), int((1.0 - rawValues[i + 1]) * plotHeight) + titleHeight + padding)

        if showSaliency:
            # Plot saliency for the current point
            cv2.circle(dataPlot, currentPoint, 25, (1.0, 1.0 - saliency[i], 1.0 - saliency[i], 1.0), -1)

            # Fill the region with saliency values
            # TODO: Fill the pixel values directly using lines (smooth boundary)
            if i < rawValues.shape[0] - 1:
                heightDiff = nextPoint[1] - currentPoint[1]
                widthDiff = nextPoint[0] - currentPoint[0]
                saliencyDiff = saliency[i + 1] - saliency[i]
                heightStride = heightDiff / widthDiff
                saliencyStride = saliencyDiff / widthDiff

                numStepsToCompensate = abs(int(np.round(heightStride))) - 1
                # print ("Compensating using %d steps (%f height stride)" % (numStepsToCompensate, heightStride))

                pointHeight = currentPoint[1]
                pointSaliency = saliency[i]
                for it in range(currentPoint[0] + 1, nextPoint[0]):
                    pointHeight += heightStride
                    pointSaliency += saliencyStride
                    # cv2.circle(dataPlot, (it, int(pointHeight)), 25, (1.0, 0.0, 0.0, pointSaliency), -1)
                    cv2.circle(dataPlot, (it, int(pointHeight)), 25,
                               (1.0, 1.0 - pointSaliency, 1.0 - pointSaliency, 1.0), -1)

                    # Smoothen out if height stride is greater than 1.5
                    for step in range(1, numStepsToCompensate + 1):
                        cv2.circle(dataPlot, (it, int(pointHeight) + int(step * np.sign(heightStride))), 25,
                                   (1.0, 1.0 - pointSaliency, 1.0 - pointSaliency, 1.0), -1)

        if i < rawValues.shape[0] - 1:
            cv2.line(dataPlot, currentPoint, nextPoint, color[channelIdx] if matPlotLibStyle else (1.0, 0.6, 0.0, 1.0), 5)

        if not matPlotLibStyle:
            # Plot the point
            if plotForecast and (i == rawValues.shape[0] - 1):
                cv2.circle(dataPlot, currentPoint, 15, (1.0, 0.0, 0.0, 1.0), -1)
            else:
                cv2.circle(dataPlot, currentPoint, 10, (0.0, 0.0, 0.0, 1.0), -1)

    if matPlotLibStyle:
        for i in range(allValues.shape[0]):
            for j in range(allValues.shape[1]):
                rawValues = allValues[i, :]
                drawPoint(j, channelIdx=i)
    else:
        for i in range(rawValues.shape[0]):
            drawPoint(i)

    if showFilterCluster:
        clusterBarWidth = int(filterWidth / 20)
        dataPlotNew = np.ones((filterHeight, filterWidth + clusterBarWidth, 4), dtype=np.float32)

        dataPlotNew[:, :clusterBarWidth, :] = currentClusterColor
        cv2.rectangle(dataPlotNew, (borderStride, borderStride), (clusterBarWidth + borderStride, filterHeight - borderStride),
                      (0.0, 0.0, 0.0, 1.0), borderWidth)

        # Add the rest of the plot
        dataPlotNew[:, clusterBarWidth:, :] = dataPlot

        return dataPlotNew

    return dataPlot


def writePlotToFile(dataPlot, fileName):
    # Convert image from RGBA to BGRA
    rgba = np.split(dataPlot, indices_or_sections=4, axis=-1)
    bgra = np.concatenate([rgba[2], rgba[1], rgba[0], rgba[3]], axis=-1)
    cv2.imwrite(fileName, bgra * 255.0)


# Can either create a grid view of a stacked view
def createCombinedPlot(dataPlots, gridView=False, filtersInfo=None, maxPlotsPerRow=-1):
    filterHeight = dataPlots[1].shape[0]
    filterWidth = dataPlots[1].shape[1]
    numPlots = int(len(dataPlots) / 2)

    plotMargin = int(filterHeight / 15) # Only for grid view

    # If filter information is available, sort the filters in cluster and then w.r.t. importance within each cluster
    filterIDs = None
    if filtersInfo is not None:
        assert len(filtersInfo) == numPlots, "Error: Number of plots and filter info is not aligned"

        # Sort filters in clusters
        filterClusters = {}
        for i in range(len(filtersInfo)):
            # Sort the filters
            filterImportance, filterCluster = filtersInfo[i]
            if filterCluster not in filterClusters:
                filterClusters[filterCluster] = []
            filterClusters[filterCluster].append((i, filterImportance))

        # Sort filters w.r.t. maximum importance within a cluster
        filterIDs = []
        for cluster in filterClusters:
            filterID = [x[0] for x in filterClusters[cluster]]
            filterImportance = [x[1] for x in filterClusters[cluster]]

            sorted = np.argsort(filterImportance)[::-1] # Sort in descending order
            filterIDs = filterIDs + [filterID[x] for x in sorted]

    if gridView:
        # Grid view
        if maxPlotsPerRow > 0:
            numPlotsPerRow = maxPlotsPerRow
            numPlotsPerCol = int(np.ceil(numPlots / maxPlotsPerRow))
        else:
            numPlotsPerRow = int(np.ceil(np.sqrt(numPlots)))
            numPlotsPerCol = numPlotsPerRow

        filterStrideH = filterHeight + plotMargin
        filterStrideW = filterWidth + plotMargin

        combinedPlotH = (filterStrideH * numPlotsPerCol) - plotMargin
        combinedPlotW = (filterStrideW * numPlotsPerRow) - plotMargin

        print("Total Plots: %d | Number of plots per row: %d | Number of plots per column: %d" % (numPlots, numPlotsPerRow, numPlotsPerCol))

    else:
        # Stacked view
        filterStrideH = int(filterHeight / 8)
        filterStrideW = filterStrideH

        combinedPlotH = filterHeight + (filterStrideH * (numPlots - 1))
        combinedPlotW = filterWidth + (filterStrideW * (numPlots - 1))

    print("Final plot dims: (%d, %d, %d)" % (combinedPlotH, combinedPlotW, 4))
    completeDataPlot = np.ones((combinedPlotH, combinedPlotW, 4))
    completeDataPlot[:, :, -1] = 0.0 # Everything is transparent initially

    row = -1
    for i in range(numPlots):
        if filterIDs is not None:
            plot = dataPlots[int(filterIDs[i] * 2) + 1]
        else:
            plot = dataPlots[int(i * 2) + 1]
        alpha = np.expand_dims(plot[:, :, -1], axis=-1)

        if gridView:
            if i % numPlotsPerRow == 0:
                row += 1
                col = 0

            background = completeDataPlot[(row * filterStrideH):(row * filterStrideH + filterHeight),
                         (col * filterStrideW):(col * filterStrideW + filterWidth), :]

            completeDataPlot[(row * filterStrideH):(row * filterStrideH + filterHeight),
                (col * filterStrideW):(col * filterStrideW + filterWidth), :] = \
                (np.multiply(background, 1.0 - alpha) + np.multiply(plot, alpha))

            col += 1

        else:
            background = completeDataPlot[(i * filterStrideH):(i * filterStrideH + filterHeight),
                         (i * filterStrideW):(i * filterStrideW + filterWidth), :]

            completeDataPlot[(i * filterStrideH):(i * filterStrideH + filterHeight),
                (i * filterStrideW):(i * filterStrideW + filterWidth), :] = \
                (np.multiply(background, 1.0 - alpha) + np.multiply(plot, alpha))

    # Make the background transparent (Problem: Also masks out the name and the saliency)
    # mask = np.all(completeDataPlot[:, :] == np.array([1.0, 1.0, 1.0, 1.0]), axis=-1)
    # completeDataPlot[mask] = np.array([1.0, 1.0, 1.0, 0.0])

    return completeDataPlot


# Plot input data
def plotInputData(useSaliency, saliencyMap=None):
    inputData = np.array(data[inputLayerIdx])

    if saliencyMap is not None:
        dataPlots = []

    for idx, filterName in enumerate(inputFeatureNames):
        inputFeature = inputData[idx, :]
        # print (inputFeature)
        if saliencyMap is None:
            saliency = np.array(data[-1][0][saliencyMapIdx])[idx, :]
        else:
            saliency = saliencyMap[idx, :]

        plotForecast = False
        if forecastingTask and (idx == 0):  # Considering only the first channel is the original signal
            forecast = data[-1][0][rawValueIdx]
            inputFeature = np.append(inputFeature, [forecast])
            saliency = np.append(saliency, [0.0])
            plotForecast = True

        dataPlot = createPlotImg(filterName=filterName, filterImportance=0.8, filterCluster=0, maxFilterCluster=0,
                                 rawValues=inputFeature, saliency=saliency, plotForecast=plotForecast,
                                 showFilterImportance=False, showSaliency=useSaliency, showFilterCluster=False)

        if saliencyMap is not None:
            # Return the plots when plotting against custom saliency maps
            dataPlots.append(filterName)
            dataPlots.append(dataPlot)
        else:
            # Only write plots to file when plotting the saliency w.r.t. output
            fileName = os.path.join(outputDirectory, filterName + ("-saliency" if useSaliency else "") + ".png")
            writePlotToFile(dataPlot, fileName)

    if saliencyMap is not None:
        completeDataPlot = createCombinedPlot(dataPlots)
        fileName = os.path.join(outputDirectory, "combined.png")
        writePlotToFile(completeDataPlot, fileName)

        return dataPlots


# Plot layer data
def plotLayer(layerID, useSaliency, useImportance, useCluster, numFilters=3, plotInputSaliency=False, createCombinedLayerPlot=False):
    layers = [layerID]
    if layerID == -1:
        layers = list(range(dataStartingIdx, len(data)))

    for layerID in layers:
        layerData = data[dataStartingIdx + layerID]
        # for filterIdx in range(len(layerData)):
        assert numFilters <= len(layerData), "Number of filters cannot be greater than the total number of filters in the layer (%d > %d)" % (numFilters, len(layerData))
        if numFilters == -1:
            numFilters = len(layerData)
        filterImportance = np.array([layerData[i][filterImportanceIdx] for i in range(len(layerData))])
        topFilterIdx = np.argsort(filterImportance)[::-1]
        topFilterIdx = topFilterIdx[:numFilters]
        maxClusterIdx = max([layerData[i][filterClusterIdx] for i in range(len(layerData))])
        for filterIdx in topFilterIdx:
            filterName = data[layerNamesIdx][layerID] + "-" + str(filterIdx)
            rawValues = np.array(layerData[filterIdx][rawValueIdx])
            saliency = np.array(layerData[filterIdx][filterSaliencyIdx])
            filterImportance = float(layerData[filterIdx][filterImportanceIdx])
            filterCluster = int(layerData[filterIdx][filterClusterIdx])
            dataPlot = createPlotImg(filterName=filterName, filterImportance=filterImportance, filterCluster=filterCluster, maxFilterCluster=maxClusterIdx, rawValues=rawValues,
                                     saliency=saliency, showFilterImportance=useImportance, showSaliency=useSaliency, showFilterCluster=useCluster)

            fileName = os.path.join(outputDirectory, filterName + ("-saliency" if useSaliency else "") + ("-importance" if useImportance else "") + ("-cluster" if useCluster else "") + ".png")
            writePlotToFile(dataPlot, fileName)

            if plotInputSaliency:
                # Plot the input saliency map
                inputSaliency = np.array(layerData[filterIdx][saliencyMapIdx])
                dataPlots = plotInputData(useSaliency=True, saliencyMap=inputSaliency)

                for i in range(0, len(dataPlots), 2):
                    fileName = os.path.join(outputDirectory, filterName + "-input-" + dataPlots[i] + ".png")
                    writePlotToFile(dataPlots[i+1], fileName)

        if createCombinedLayerPlot:
            # Write combined filter clusters
            for currentTuple in [(False, False, False, False, False), (False, True, False, False, False), (False, True, True, False, False), (True, True, True, True, False), (True, True, True, True, True)]:
                plotCluster = currentTuple[0]
                showImportance = currentTuple[1]
                showSaliency = currentTuple[2]
                sortFilters = currentTuple[3]
                gridView = currentTuple[4]

                allFilters = []
                allFiltersInfo = []
                if sortFilters:
                    filterImportance = np.array([layerData[i][filterImportanceIdx] for i in range(len(layerData))])
                    filterList = np.argsort(filterImportance) # Filters are stacked in the inverse order (so sorting in ascending order is desired)
                else:
                    filterList = range(len(layerData))

                for filterIdx in filterList:
                    filterName = data[layerNamesIdx][layerID] + "-" + str(filterIdx)
                    rawValues = np.array(layerData[filterIdx][rawValueIdx])
                    saliency = np.array(layerData[filterIdx][filterSaliencyIdx])
                    filterImportance = float(layerData[filterIdx][filterImportanceIdx])
                    filterCluster = int(layerData[filterIdx][filterClusterIdx])
                    dataPlot = createPlotImg(filterName=filterName, filterImportance=filterImportance,
                                             filterCluster=filterCluster, maxFilterCluster=maxClusterIdx,
                                             rawValues=rawValues, saliency=saliency, showFilterImportance=showImportance,
                                             showSaliency=showSaliency, showFilterCluster=plotCluster)

                    if RESIZE_RATIO != 1.0:
                        dataPlot = cv2.resize(dataPlot, None, fx=RESIZE_RATIO, fy=RESIZE_RATIO) # Resize the image to the defined ratio
                    allFilters.append(filterName)
                    allFilters.append(dataPlot)
                    allFiltersInfo.append([filterImportance, filterCluster])

                combinedPlot = createCombinedPlot(allFilters, gridView=gridView, filtersInfo=allFiltersInfo, maxPlotsPerRow=-1)
                # combinedPlot = cv2.resize(combinedPlot, None, fx=0.25, fy=0.25) # Resize the image to one-fourth of its dimensions
                fileName = os.path.join(outputDirectory, data[layerNamesIdx][layerID] + "-" + ("cluster-" if plotCluster else "") +
                                        ("imp-" if showImportance else "") + ("sort-" if sortFilters else "") +
                                        ("sal-" if showSaliency else "") + ("grid-" if gridView else "") + "combined.png")
                writePlotToFile(combinedPlot, fileName)

            # Create combined plot for most prominent filters from a cluster
            allFilters = []
            importantFilters = {}
            for filterIdx in range(len(layerData)):
                filterName = data[layerNamesIdx][layerID] + "-" + str(filterIdx)
                rawValues = np.array(layerData[filterIdx][rawValueIdx])
                saliency = np.array(layerData[filterIdx][filterSaliencyIdx])
                filterImportance = float(layerData[filterIdx][filterImportanceIdx])
                filterCluster = int(layerData[filterIdx][filterClusterIdx])

                if filterCluster not in importantFilters:
                    importantFilters[filterCluster] = [filterName, rawValues, saliency, filterImportance, filterCluster]
                else:
                    oldFilterImportance = importantFilters[filterCluster][-2]
                    if oldFilterImportance < filterImportance:
                        importantFilters[filterCluster] = [filterName, rawValues, saliency, filterImportance, filterCluster]

            for cluster in importantFilters:
                filterName, rawValues, saliency, filterImportance, filterCluster = importantFilters[cluster]
                dataPlot = createPlotImg(filterName=filterName, filterImportance=filterImportance,
                                         filterCluster=filterCluster, maxFilterCluster=maxClusterIdx,
                                         rawValues=rawValues, saliency=saliency, showFilterImportance=showImportance,
                                         showSaliency=showSaliency, showFilterCluster=plotCluster)

                if RESIZE_RATIO != 1.0:
                    dataPlot = cv2.resize(dataPlot, None, fx=RESIZE_RATIO,
                                          fy=RESIZE_RATIO)  # Resize the image to the defined ratio
                allFilters.append(filterName)
                allFilters.append(dataPlot)

            combinedPlot = createCombinedPlot(allFilters)
            fileName = os.path.join(outputDirectory, data[layerNamesIdx][layerID] + "-pruning-combined.png")
            writePlotToFile(combinedPlot, fileName)


def plotInverseOptimizationAndAdversarialExamples():
    # Get the inverse optimization output
    inverseOptimizationOutput = data[inverseOptimizationIdx]
    inverseOptimizationStart = np.array(inverseOptimizationOutput[0])
    inverseOptimizationStartPrediction = np.array(inverseOptimizationOutput[1])
    inverseOptimizationStartSaliencyMap = np.array(inverseOptimizationOutput[2])
    inverseOptimizationRawOutput = np.array(inverseOptimizationOutput[3])
    inverseOptimizationPrediction = inverseOptimizationOutput[4]
    inverseOptimizationSaliencyMap = np.array(inverseOptimizationOutput[5])

    print("Starting series shape:", inverseOptimizationStart.shape,
          "| Starting series prediction:", inverseOptimizationStartPrediction,
          "| Starting series saliency shape:", inverseOptimizationStartSaliencyMap.shape)

    print("Inverse optimization shape:", inverseOptimizationRawOutput.shape,
          "| Inverse optimization prediction:", inverseOptimizationPrediction,
          "| Inverse optimization saliency shape:", inverseOptimizationSaliencyMap.shape)

    # Plot the starting series
    for i in range(len(inputFeatureNames)):
        inputFeature = inverseOptimizationStart[0, :, i]
        saliency = inverseOptimizationStartSaliencyMap[0, :, i]

        plotForecast = False
        if forecastingTask and (i == 0):  # Considering only the first channel is the original signal
            forecast = inverseOptimizationStartPrediction[0]
            inputFeature = np.append(inputFeature, [forecast])
            saliency = np.append(saliency, [0.0])
            plotForecast = True

        dataPlot = createPlotImg(filterName=inputFeatureNames[i] + " - InvOpt", filterImportance=0.0, filterCluster=0,
                                 maxFilterCluster=0, rawValues=inputFeature, saliency=saliency, plotForecast=plotForecast,
                                 showFilterImportance=False, showSaliency=True, showFilterCluster=False)

        fileName = os.path.join(outputDirectory, "invOpt-start-" + inputFeatureNames[i] + ".png")
        writePlotToFile(dataPlot, fileName)

    # Plot the final series
    for i in range(len(inputFeatureNames)):
        inputFeature = inverseOptimizationRawOutput[0, :, i]
        saliency = inverseOptimizationSaliencyMap[0, :, i]

        plotForecast = False
        if forecastingTask and (i == 0):  # Considering only the first channel is the original signal
            forecast = inverseOptimizationPrediction[0]
            inputFeature = np.append(inputFeature, [forecast])
            saliency = np.append(saliency, [0.0])
            plotForecast = True

        dataPlot = createPlotImg(filterName=inputFeatureNames[i] + " - InvOpt", filterImportance=0.0, filterCluster=0,
                                 maxFilterCluster=0, rawValues=inputFeature, saliency=saliency, plotForecast=plotForecast,
                                 showFilterImportance=False, showSaliency=True, showFilterCluster=False)

        fileName = os.path.join(outputDirectory, "invOpt-" + inputFeatureNames[i] + ".png")
        writePlotToFile(dataPlot, fileName)

    # Get the adversarial examples output
    adversarialExamplesOutput = data[adversarialExamplesIdx]
    adversarialExamplesRawOutput = np.array(adversarialExamplesOutput[0])
    adversarialExamplesPrediction = adversarialExamplesOutput[1]
    adversarialExamplesSaliencyMap = np.array(adversarialExamplesOutput[2])
    print("Adversarial example shape:", adversarialExamplesRawOutput.shape,
          "| Adversarial example prediction:", adversarialExamplesPrediction,
          "| Adversarial example saliency shape:", adversarialExamplesSaliencyMap.shape)

    for i in range(len(inputFeatureNames)):
        inputFeature = adversarialExamplesRawOutput[0, :, i]
        saliency = adversarialExamplesSaliencyMap[0, :, i]

        plotForecast = False
        if forecastingTask and (i == 0):  # Considering only the first channel is the original signal
            forecast = adversarialExamplesPrediction[0]
            inputFeature = np.append(inputFeature, [forecast])
            saliency = np.append(saliency, [0.0])
            plotForecast = True

        dataPlot = createPlotImg(filterName=inputFeatureNames[i] + " - AdvEx", filterImportance=0.0, filterCluster=0,
                                 maxFilterCluster=0, rawValues=inputFeature, saliency=saliency, plotForecast=plotForecast,
                                 showFilterImportance=False, showSaliency=True, showFilterCluster=False)

        fileName = os.path.join(outputDirectory, "advEx-" + inputFeatureNames[i] + ".png")
        writePlotToFile(dataPlot, fileName)


# Main function
if __name__ == "__main__":
    completePlotting = True
    batchMode = False
    exampleIdx = 60

    url = "http://pc-4133:5000/viz/api/"
    layerID = 1  # Second conv layer

    # Plot the examples
    rootDirectory = "./PresentationPlots/"
    if os.path.exists(rootDirectory):
        shutil.rmtree(rootDirectory)
    os.mkdir(rootDirectory)

    # Switch to the test set (automatically resets the iterator)
    response = requests.get(url + "test")
    if response.status_code != 200:
        print("Error: Unable to switch to the test set!")
        exit(-1)

    if batchMode:
        exampleIdx = 0
    else:
        response = requests.get(url + "set?iterator=" + str(exampleIdx))  # Set the iterator to the corresponding example
        if response.status_code != 200:
            print("Error: Unable to set the iterator!")
            exit(-1)

        outputDirectory = rootDirectory
        print("Example ID set response:", response.status_code)

    while True:
        plotExample = True

        # Create the corresponding directory for batch model
        if batchMode:
            # Fetch the data
            response = requests.get(url + "get_example")
            gt = float(json.loads(response.content.decode("utf-8"))["y"])
            plotExample = gt == 1.0

        if plotExample:
            if batchMode:
                print("Fetching example #", exampleIdx)
                outputDirectory = rootDirectory + str(exampleIdx)
                if not os.path.exists(outputDirectory):
                    os.mkdir(outputDirectory)

            # Fetch the data
            response = requests.get(url + "fetch")
            print("Example fetch response:", response.status_code)
            if response.status_code != 200:
                print("Error: Unable to retrieve data from service!")
                exit(-1)

            data = json.loads(response.content.decode("utf-8"))["data"]
            inputFeatureNames = data[inputFeaturesNamesIdx]
            classes = data[classNamesIdx]
            forecastingTask = (data[datasetTypeIdx] == "Regression")
            if forecastingTask:
                assert len(data[-1]) == 1, "Error: Multi-variable forecasting not supported at this point!"

            gt = data[groundTruthIdx]
            print("Class names:", classes, "| Input features:", inputFeatureNames)
            print("Ground Truth:", gt, "| Prediction:", data[-1][0][rawValueIdx])  # Zero denotes the filter idx (only one output class)
            assert(not batchMode or gt == 1.0)

            if completePlotting:
                print("Plot inverse optimization output")
                plotInverseOptimizationAndAdversarialExamples()

            print("Plotting input layer")
            plotInputData(useSaliency=False)
            plotInputData(useSaliency=True)

            if completePlotting:
                print("Plotting intermediate layers")
                plotLayer(layerID, useSaliency=False, useImportance=False, useCluster=False)
                plotLayer(layerID, useSaliency=False, useImportance=True, useCluster=False)
                plotLayer(layerID, useSaliency=True, useImportance=False, useCluster=False)
                plotLayer(layerID, useSaliency=True, useImportance=True, useCluster=False)
                plotLayer(layerID, useSaliency=True, useImportance=True, useCluster=True, plotInputSaliency=True, createCombinedLayerPlot=True)

            # cv2.imshow("Output", bgra)
            # keyPressed = cv2.waitKey(0)
            # if keyPressed == ord('q'):
            #   print ("Process terminated!")
            #   exit(-1)

        if not batchMode:
            break
        else:
            # Load next example
            response = requests.get(url + "next")
            if json.loads(response.content.decode("utf-8"))["status"] != "ok":
                break
            exampleIdx += 1
