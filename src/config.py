from enum import Enum


class Dataset(Enum):
    # Indices
    TRAIN = 0
    TEST = 1

    # Dataset types
    CLASSIFICATION = 0
    REGRESSION = 1

    # Dataset names
    INTERNET_TRAFFIC = 0
    ANOMALY_DETECTION = 1


class Clustering(Enum):
    # Clustering types
    K_MEANS = 0
    ADAPTIVE_K_MEANS = 1
    GMM = 2
    MEAN_SHIFT = 3
    HIERARCHICAL = 4

    # Distance metrics
    DTW = 0
    EUCLIDEAN = 1

    # Automatic number of cluster selection methods
    ASANKA = 0
    SILHOUETTE = 1


# Dataset settings
DATASET = Dataset.ANOMALY_DETECTION
DATASET_TYPE = Dataset.REGRESSION if DATASET == Dataset.INTERNET_TRAFFIC else Dataset.CLASSIFICATION
CLASS_NAMES = ["" if DATASET == Dataset.INTERNET_TRAFFIC else "Anomaly"]
BATCH_NORM = False  # Takes more time - new layers
INPUT_FEATURE_NAMES = ["Internet Traffic", "1D Derivative"] if DATASET == Dataset.INTERNET_TRAFFIC else ["Pressure", "Temperature", "Torque"] if DATASET == Dataset.ANOMALY_DETECTION else ["Signal"]


CLUSTERING_METHOD = Clustering.HIERARCHICAL
DISTANCE_METRIC = Clustering.DTW if CLUSTERING_METHOD == Clustering.HIERARCHICAL else Clustering.EUCLIDEAN
SELECTION_TYPE = Clustering.SILHOUETTE
MEAN_SHIFT_BANDWIDTH = 0.5

SCALE_X_AXIS = True
SCALE_X_AXIS = False if DISTANCE_METRIC == Clustering.DTW else SCALE_X_AXIS  # X-axis scaling should be turned off with DTW
RANDOM_STATE = 313  # or None
