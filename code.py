import cv2
import math 
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
plt.style.use('dark_background')

# This function is used to show the images

def showImgs(imgs, n_imgs, i_imgs):
   n = sqrt(n_imgs)
   m = n
   p = 1
   if n != int(n):
      n = int(n)
      m = n + 1
   print("n, m", n, m)
   fig = plt.figure()
   for i in i_imgs:
      fig.add_subplot(int(n), int(m), p)
      plt.imshow(imgs[i], cmap='gray')
      plt.axis('off')
      p += 1
plt.show()


# import test images for the algo 


faces = fetch_olivetti_faces()
images = faces.images
images.shape

features = faces.data  # features
targets = faces.target # targets
showImgs(images, 100, range(100))

# ...

# Show the images and corresponding target labels
def showImgsWithLabels(imgs, n_imgs, i_imgs, labels):
    n = sqrt(n_imgs)
    m = n
    p = 1
    if n != int(n):
        n = int(n)
        m = n + 1
    fig = plt.figure()
    for i in i_imgs:
        fig.add_subplot(int(n), int(m), p)
        plt.imshow(imgs[i], cmap='gray')
        plt.axis('off')
        plt.title(f"Person {labels[i]}")
        p += 1
    plt.show()

# ...

# Modify this line to use showImgsWithLabels
showImgsWithLabels(images, 100, range(100), targets)

# Perform PCA for dimensionality reduction
n_components = 150  # You can adjust this based on your requirements
pca = PCA(n_components=n_components, whiten=True).fit(features)
eigenfaces = pca.components_.reshape((n_components, 64, 64))  # Assuming the images are 64x64 pixels

# Visualize the Eigenfaces
def plotEigenfaces(eigenfaces, n_eigenfaces):
    plt.figure(figsize=(15, 10))
    for i in range(n_eigenfaces):
        plt.subplot(10, 15, i + 1)
        plt.imshow(eigenfaces[i], cmap='gray')
        plt.axis('off')
        plt.title(f'Eigenface {i + 1}')
    plt.show()

# Visualize the first 30 Eigenfaces
plotEigenfaces(eigenfaces, n_eigenfaces=30)


import cv2
import math
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# Set up matplotlib configurations for plotting
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
plt.style.use('dark_background')

# Function to display images with labels
def showImgsWithLabels(imgs, n_imgs, i_imgs, labels):
    n = sqrt(n_imgs)
    m = n
    p = 1
    if n != int(n):
        n = int(n)
        m = n + 1
    fig = plt.figure()
    for i in i_imgs:
        fig.add_subplot(int(n), int(m), p)
        plt.imshow(imgs[i], cmap='gray')
        plt.axis('off')
        plt.title(f"Person {labels[i]}")
        p += 1
    plt.show()

# Load the Olivetti Faces dataset
faces = fetch_olivetti_faces()
images = faces.images
features = faces.data  # Features
targets = faces.target  # Target labels

# Display the first 100 images with labels
showImgsWithLabels(images, 100, range(100), targets)

# Perform PCA for dimensionality reduction
n_components = 150  # You can adjust this based on your requirements
pca = PCA(n_components=n_components, whiten=True).fit(features)
X_pca = pca.transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, targets, test_size=0.25, random_state=42)

# Train a K-Nearest Neighbors (KNN) classifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print classification report to evaluate the model
print(classification_report(y_test, y_pred, target_names=[f"Person {i}" for i in range(40)]))


# Get the unique class labels from y_test
unique_labels = np.unique(y_test)

# Create target_names based on unique_labels
target_names = [f"Person {label}" for label in unique_labels]

# Print classification report with the corrected target_names
print(classification_report(y_test, y_pred, target_names=target_names))

