import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import time
from numpy.linalg import norm

from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import ResNet50
import keras.utils as image

import warnings
warnings.filterwarnings("ignore")

model = ResNet50(weights = "imagenet", include_top = False, input_shape = (244, 244, 3))

def feature_extraction(img_path, model):
    input_shape = (244, 244, 3)
    img = image.load_img(img_path, target_size = (input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features/norm(flattened_features)
    return normalized_features

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list

root_dir = 'Data/Training'
test_dir = 'Data/Testing'
filenames = sorted(get_file_list(root_dir))
test_file = sorted(get_file_list(test_dir))

feature_list = []
for i in tqdm_notebook(range(len(filenames))):
    feature_list.append(feature_extraction(filenames[i], model))

test_feature = []
for j in tqdm_notebook(range(len(test_file))):
    test_feature.append(feature_extraction(test_file[j], model))

image_index = 0
neighbors = NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'matching').fit(feature_list)
distances, indices = neighbors.kneighbors([test_feature[image_index]])

plt.imshow(mpimg.imread(test_file[image_index]))

fig = plt.figure(figsize=(10,10))
rows = 2
columns = 6
fig.add_subplot(rows, columns, 1)
plt.imshow(mpimg.imread(test_file[image_index]))
plt.axis('off')
plt.title('Query Image')

print('Similar Matches')
indices = indices.flatten()
print(indices)

for i in range(len(indices)):
  fig.add_subplot(rows, columns, i+2)

  plt.imshow(mpimg.imread(filenames[indices[i]]))
  plt.axis('off')
  plt.title('match_'+str(i+1)+'_image_id'+str(indices[i]))