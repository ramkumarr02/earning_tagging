import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, normalize

from sklearn import preprocessing, linear_model, naive_bayes, metrics, svm, ensemble

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer

import xgboost

from tqdm import tqdm
import plotly.express as px

from matplotlib import pyplot as plt

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy

import collections

print(f" Found and Using {len(tensorflow.config.experimental.list_physical_devices('GPU'))} GPU")


import warnings
warnings.filterwarnings('ignore')