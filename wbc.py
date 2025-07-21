#!/usr/bin/env python3
# -- coding: utf-8 --
"""
@author: chandra
"""


from _future_ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.mobilenet_v2 import MobileNetV2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.applications.xception import Xception




SIZE = 128
train_images = []
train_labels = [] 
for directory_path in glob.glob("/home/amit/data sets/PBC_dataset_normal_DIB_224/*"):
    label = directory_path.split("/")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)






x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test = (train_images, test_images, train_labels,test_labels)

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

label_encoder = LabelEncoder()

# Fit and transform the y_train labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Add, Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Lambda, Reshape, Multiply, MaxPooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Lambda, Reshape, Multiply
import tensorflow as tf

def sk_block2(input, filters, strides=1):
    # Branch 1
    branch1 = Conv2D(filters, (3, 3), strides=strides, padding="same")(input)
    branch1 = BatchNormalization()(branch1)
    branch1 = ReLU()(branch1)
    branch1 = Conv2D(filters, (3, 3), strides=1, padding="same")(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = ReLU()(branch1)

    # Branch 2
    branch2 = Conv2D(filters, (5, 5), strides=strides, padding="same", dilation_rate=3)(input)
    branch2 = BatchNormalization()(branch2)
    branch2 = ReLU()(branch2)
    branch3 = Conv2D(filters, (5, 5), strides=1, padding="same", dilation_rate=5)(branch2)
    branch3 = BatchNormalization()(branch3)
    branch3 = ReLU()(branch3)
    branch4 = Add()([branch2, branch3])

    fused = Add()([branch1, branch4])

    attention = GlobalAveragePooling2D()(fused)
    x_a = Dense(filters // 16, activation='relu')(attention)
    x_a = Dense(2 * filters, activation='sigmoid')(x_a)

    def split_attention(x):
        return tf.split(x, num_or_size_splits=2, axis=1)
    
    def split_attention_output_shape(input_shape):
        filters = input_shape[-1] // 2
        return [input_shape[:-1] + (filters,), input_shape[:-1] + (filters,)]

    a1, a2 = Lambda(split_attention, output_shape=split_attention_output_shape)(x_a)
    a1 = Reshape((1, 1, filters))(a1)
    a2 = Reshape((1, 1, filters))(a2)
    print(a1.shape)
    print(a2.shape)

    output = Add()([Multiply()([branch1, a1]), Multiply()([branch4, a2])])
    return output






def CAB_Block(feature_map):
  x1 = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(feature_map)
  x1 = tf.keras.activations.gelu(x1, approximate=False)

  x3 = tf.keras.layers.DepthwiseConv2D(5,padding="same")(x1)

  x3 = tf.keras.layers.DepthwiseConv2D(7,dilation_rate = (3,3),padding="same")(x3)

  Att = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(x3)

  XL = tf.keras.layers.multiply([x1,Att])

  XL = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(XL)

  XL = tf.keras.activations.gelu(XL, approximate=False)

  X_final = tf.keras.layers.Add()([XL, feature_map])

  x2 = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(X_final)

  x2 = tf.keras.layers.DepthwiseConv2D(3,padding="same")(x2)

  x1 = tf.keras.activations.gelu(x1, approximate=False)

  x2 = tf.keras.layers.Conv2D(feature_map.shape[-1],1,padding="same")(x2)

  final_output = tf.keras.layers.Add()([x2,X_final])

  return final_output


def multiScale_feature_fusion_block(sampled_dense_output1,sampled_dense_output2,sampled_dense_output3,sampled_dense_output4):
  concat_layer = tf.keras.layers.Concatenate()([sampled_dense_output1,sampled_dense_output2,sampled_dense_output3,sampled_dense_output4])
  convolution_layer = tf.keras.layers.Conv2D(128,1,padding="same")(concat_layer)
  x1 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output1])
  x2 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output2])
  x3 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output3])
  x4 = tf.keras.layers.Concatenate()([convolution_layer, sampled_dense_output4])
  
  return [x1,x2,x3,x4]



input_shape = (128,128,3)
input_tensor = Input(shape = input_shape)
#base_model = Xception(weights ='imagenet',include_top = False,input_tensor= input_tensor)
# base_model = DenseNet201(weights ='imagenet',include_top = False,input_tensor= input_tensor)
base_model = MobileNetV2(weights ='imagenet',include_top = False,input_tensor= input_tensor)

# Make all the layers in model_2_base_model trainable
base_model.trainable = False

# # Freeze all layers except for the last 10
# for layer in base_model.layers[:-10]:
#   layer.trainable = False
  
# feature1 = base_model.get_layer('block3_sepconv2').output
# feature2 = base_model.get_layer('block4_sepconv2').output
# feature3 = base_model.get_layer('block13_sepconv2').output

# feature1 = base_model.get_layer('conv3_block12_concat').output
# feature2 = base_model.get_layer('conv4_block48_concat').output
# feature3 = base_model.get_layer('conv5_block32_concat').output


base_model.summary()
feature1 = base_model.get_layer('block_3_expand').output
feature2 = base_model.get_layer('block_6_expand').output
feature3 = base_model.get_layer('block_13_expand').output

# feature1 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(feature1)

print( feature1.shape)
print(feature2.shape)
print(feature3.shape)




# output of the AcsConv block
# AcsConv_layer = AcsConv_block(feature3)

# AcsConv_layer1 = sk_block1(feature3,64)
# AcsConv_layer.shape

newfeature1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(feature1)
# newfeature2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature2)
newfeature3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature3)
#newAcsConv_layer = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(AcsConv_layer)


AcsConv_layer4=sk_block2(newfeature3,64)

print(AcsConv_layer4.shape)
print(newfeature1.shape)
print(newfeature3.shape)








x1, x2, x3 , x4 = multiScale_feature_fusion_block(newfeature1,feature2,newfeature3,
AcsConv_layer4)
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)







#multiscalce_feature_fusion_list = multiScale_feature_fusion_block(newfeature1,feature2,newfeature3,newAcsConv_layer)
#applying Context attention block
cs1 = tf.keras.layers.Conv2D(128,1,padding="same")(x1)
print(cs1.shape)
cs1 = CAB_Block(cs1)
cs2 = tf.keras.layers.Conv2D(128,1,padding="same")(x2)
cs2 = CAB_Block(cs2)
cs3 = tf.keras.layers.Conv2D(128,1,padding="same")(x3)
cs3 = CAB_Block(cs3)
cs4 = tf.keras.layers.Conv2D(128,1,padding="same")(x4)
cs4 = CAB_Block(cs4)


print(cs1.shape)
print(cs2.shape)
print(cs3.shape)
print(cs4.shape)


# AcsConv_layer1=sk_block2(cs1,128)
# AcsConv_layer2 = sk_block2(cs2,128)
# AcsConv_layer3=sk_block2(cs3,128)



# print(AcsConv_layer1.shape)
# print(AcsConv_layer2.shape)
# print(AcsConv_layer3.shape)


# Cs1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(cs1)
# # newfeature2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature2)
# Cs3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(cs3)

# print(Cs1.shape)
# print(cs2.shape)
# print(Cs3.shape)

final_output=tf.keras.layers.add([cs1,cs2,cs3,cs4])
print(final_output.shape)



# adding all the attention layers

final_conv=tf.keras.layers.Conv2D(128,1,padding="same")(final_output)
print(final_conv.shape)

final_pooled_output = tf.keras.layers.GlobalAveragePooling2D()(final_conv)
# flattened_output = tf.keras.layers.Flatten()(final_pooled_output)
flattened_output = tf.keras.layers.Dense(256,activation="relu")(final_pooled_output)
final_dense_layer = tf.keras.layers.Dense(5, activation='softmax', name='output_layer')(flattened_output)

model = Model(inputs=input_tensor, outputs=final_dense_layer)

model.summary()


model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])



checkpoint_path = "/home/amit/data sets/results/at_batch_size=32.fC.keras" 

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_best_only=True, 
                                                         monitor='val_accuracy',
                                                         save_freq="epoch",


from tensorflow.keras.preprocessing.image import ImageDataGenerator                                                        verbose=1)
datagen = ImageDataGenerator(
       rotation_range=20,
       zoom_range=0.1,
       width_shift_range=0.1
       height_shift_range=0.1
       brightness_range=[0.8,1.2]
       horizontal_flip=True,
       vertical_flip=True,
      )

datagen.fit(x_train)

history = model.fit(datagen.flow(x_train,
                    y_train_one_hot,
                    batch_size = 32),
                    epochs  = 50,
                    validation_data = (x_test, y_test_one_hot),
                    callbacks=[checkpoint_callback])


 

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss', fontsize=30)
plt.xlabel('Epochs', fontsize=30)
plt.ylabel('Loss', fontsize=30)
plt.legend(fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()





import matplotlib.pyplot as plt
import seaborn as sns

# Use a clean style
sns.set_style("whitegrid")

# Extract loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Create a figure with a custom size
plt.figure(figsize=(12, 8))

# Plot training and validation loss
plt.plot(epochs, loss, 'o-', color='mediumseagreen', label='Training Loss', linewidth=2, markersize=6)
plt.plot(epochs, val_loss, 's-', color='crimson', label='Validation Loss', linewidth=2, markersize=6)

# Add title and axis labels
plt.title('Training and Validation Loss', fontsize=28, weight='bold')
plt.xlabel('Epochs', fontsize=24)
plt.ylabel('Loss', fontsize=24)

# Add legend, grid, and style ticks
plt.legend(fontsize=20)
plt.grid(True)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Optional: set y-limit if needed
# plt.ylim(0, max(max(loss), max(val_loss)) * 1.1)

# Tight layout for clean output
plt.tight_layout()

# Optional: save the figure
# plt.savefig("loss_plot.png", dpi=300)

# Show the plot
plt.show()








acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy', fontsize=35)
plt.xlabel('Epochs', fontsize=35)
plt.ylabel('Accuracy', fontsize=35)
plt.legend(fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35 )
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

# Apply a clean style
sns.set_style("whitegrid")

# Extract accuracy values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

# Create figure with custom size
plt.figure(figsize=(12, 8))

# Plot training and validation accuracy
plt.plot(epochs, acc, 'o-', color='royalblue', label='Training Accuracy', linewidth=2, markersize=6)
plt.plot(epochs, val_acc, 's-', color='darkorange', label='Validation Accuracy', linewidth=2, markersize=6)

# Add title and labels with appropriate font sizes
plt.title('Training and Validation Accuracy', fontsize=28, weight='bold')
plt.xlabel('Epochs', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)

# Add legend, grid, and ticks
plt.legend(fontsize=20)
plt.grid(True)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Optional: set limits if needed
plt.ylim(0, 1.05)

# Tight layout to avoid clipping
plt.tight_layout()

# Optional: save figure
# plt.savefig("accuracy_plot.png", dpi=300)

# Show the plot
plt.show()








from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd

# Define the custom functions
def split_attention(x):
    return tf.split(x, num_or_size_splits=2, axis=1)

def split_attention_output_shape(input_shape):
    filters = input_shape[-1] // 2
    return [input_shape[:-1] + (filters,), input_shape[:-1] + (filters,)]

# Define custom objects
custom_objects = {
    'split_attention': split_attention,
    'split_attention_output_shape': split_attention_output_shape
}

# Load the model with custom objects
model = load_model('/home/amit/data sets/results/at_batch_size=32.f.keras', custom_objects=custom_objects)

# Now you can use the model for prediction or evaluation
# Example:
# predictions = model.predict(x_test)


# from tensorflow.keras.models import load_model
# import pandas as pd
# # Load the model with saved weights
# # model = load_model('results/ds3.keras')
# from tensorflow.keras.models import load_model

# custom_objects = {
#     'split_attention': split_attention,  # The custom Lambda function
#     'split_attention_output_shape': split_attention_output_shape  # The custom output shape function
# }

# model = load_model('path/to/your_model.keras', custom_objects=custom_objects)


# model = load_model('results/rabbin_datasets/withmulti/batch_size16.keras',safe_mode=False)

from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
Y_pred = model.predict(x_test)
y_pred_classes = np.argmax(Y_pred, axis = -1)
y_true = np.argmax(y_test_one_hot, axis = -1)
confusion_M1=pd.crosstab(y_true,y_pred_classes)
#fig= plt.figure(figsize=(10,5))
#ax1=plt.subplot(121)
sns.set(font_scale=2.0) #edited as suggested
sns.heatmap(confusion_M1, annot=True,fmt="d", cmap='Oranges')
#plt.title("Confusion Matrix")
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')







for i in range(confusion_M1.shape[0]):
    TP=confusion_M1.iloc[i,i]
    FP=confusion_M1.iloc[i,:].sum()-TP				
    FN=confusion_M1.iloc[:,i].sum()-TP
    TN=confusion_M1.sum().sum()-TP-FP-FN
    Accuracy= (TP+TN)/confusion_M1.sum().sum()
    Precision =TP/(TP+FP)
    Recall= TP/(TP+FN)
    F1_score= (2*Precision*Recall)/ (Precision + Recall)
    print(confusion_M1.index[i], Accuracy, Precision, Recall, F1_score)

pd.DataFrame(classification_report(y_true,y_pred_classes, output_dict= True)).T



#QWK
print(" ");
from sklearn.metrics import cohen_kappa_score
# Assuming y_test and prediction_RF are your true and predicted labels
weighted_kappa = cohen_kappa_score(y_true, y_pred_classes, weights='quadratic')
print(f'Weighted Kappa: {weighted_kappa:.4f}')





	#ROC CURVE
from sklearn.metrics import roc_curve, auc
from itertools import cycle
y_score = model.predict(x_test)
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
num_classes=5
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

colors = cycle(['red','blue', 'green', 'deeppink', 'darkorange'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()










import numpy as np
from scipy import interpolate
from sklearn.metrics import auc  # Correct import
import matplotlib.pyplot as plt
from itertools import cycle
n_classes = 5

# Use np.interp instead of just interp
roc_auc_scores = []
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # Changed this line

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['red','blue', 'green', 'deeppink', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    roc_auc_scores.append(roc_auc[i])

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()



# new code plot the more professional plots 
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import auc
from itertools import cycle
import seaborn as sns

# Optional: seaborn style for cleaner plots
sns.set_style("whitegrid")

# Number of classes
n_classes = 5

# Interpolation and macro-average AUC
roc_auc_scores = []
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # Interpolate TPR at common FPR

mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot setup
plt.figure(figsize=(12, 8), dpi=300)

# Micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average (AUC = {:.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=3)

# Macro-average ROC curve
plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average (AUC = {:.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=3)

# Per-class ROC curves
colors = cycle(sns.color_palette("husl", n_classes))  # Better palette
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
    roc_auc_scores.append(roc_auc[i])

# Reference diagonal
plt.plot([0, 1], [0, 1], 'k--', lw=1)

# Axis labels and plot title
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Multi-Class ROC Curve', fontsize=22, weight='bold')

# Legend and grid
plt.legend(loc="lower right", fontsize=14)
plt.grid(True)
plt.tight_layout()

# Optional: save the figure
# plt.savefig("roc_curve_multiclass.png", dpi=300)

plt.show()