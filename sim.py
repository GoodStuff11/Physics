import tensorflow as tf
import keras
import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/htpc/Documents/Programming/PycharmProjects/QuestionSimilarity/Data/SE Questions Physics.txt', error_bad_lines=False, sep='\|\ \ \ \|', engine='python')
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten
])