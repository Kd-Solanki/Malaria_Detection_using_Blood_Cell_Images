
# importing required libraries
from skimage.io import imread, imshow
from skimage.transform import resize
import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog


# loading the trained model
pickle_in = open('Malaria_Detection.pkl', 'rb')
classifier = pickle.load(pickle_in)


# define prediction function
def predict(image):
  # Loading and Pre-processing the data
  features_hog = []
  IMG_DIMS = (128,64)
  ima = imread(image)
  ima = resize(ima,IMG_DIMS)
  #calculating HOG features
  features, hog_image = hog(ima, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
  hog_features = np.reshape(features,(features.shape[0]))
  features_hog.append(hog_features)
  features_hog = np.array(hog_features)
  # get prediction
  features_hog_rs = features_hog.reshape(1,3780)
  prediction = (classifier.predict(features_hog_rs) > 0.5).astype("int32")

  if prediction == 1:
        pred = 'Positive'
  else:
        pred = 'Negative'
  return pred
  


# define image file uploader
image = st.file_uploader("Upload image here")

# define button for getting prediction
if image is not None and st.button("Get prediction"):
    # load image using PIL
    input_image = Image.open(image)

    # show image
    st.image(input_image, use_column_width=True)

    # get prediction
    pred = predict(image)

    # print results
    "the patient is malaria ", pred
