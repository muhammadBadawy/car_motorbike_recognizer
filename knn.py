# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
#use to process image in real time 
import cv2
import numpy as np
#random number generator to use when shuffling the data
from sklearn.utils import shuffle
from keras import backend as K


from keras.utils import np_utils

img_rows=128
img_cols=128
#one color
num_channel=1
num_epoch=10
# Define the number of classes
num_classes = 2
#get full path
PATH = os.getcwd()
# Define data path
data_path = PATH + '/cars'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(128,128)).flatten()
		img_data_list.append(input_img_resize)
#input_shape=(128, 128, 1) for 128x128 RGB pictures in  data_format="channels_last"
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

#####################################
 
# =============================================================================
# Assigning Labels
        
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:879]=0
labels[880:1706]=1
# =============================================================================
# labels[404:606]=2
# labels[606:]=3
# =============================================================================
	  
#names = ['cats','dogs','horses','humans']
names = ['audi','motorbike']
# convert class labels to on-hot encoding
#Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
#x,y = shuffle(img_data,Y, random_state=2)
x = img_data
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=2, stratify=labels)


# Create a k-NN classifier with 7 neighbors: knn
#knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the training data
#knn.fit(X_train,y_train)

accuracies=[]
kVals = range(1, 30, 2)
# Print the accuracy
#print(knn.score(X_test, y_test))

for k in range(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(X_train,y_train)
 
	# evaluate the model and update the accuracies list
	score = knn.score(X_test, y_test)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)
 
# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
	accuracies[i] * 100))