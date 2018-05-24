import os
import cv2
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split




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
x,y = shuffle(img_data,labels, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=labels)

#C of the error term

svc = svm.SVC( C=10, decision_function_shape='ovr').fit(X_train, y_train)

y_true, y_pred = y_test, svc.predict(X_test)
print(classification_report(y_true, y_pred))

print('Model Acc ')
print(svc.score(X_test, y_test))