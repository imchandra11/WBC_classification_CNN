from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import ResNet50V2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import glob
import numpy as np
# from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder



SIZE = 192
train_images = []
train_labels = [] 
for directory_path in glob.glob("D:/Ayush/Rabbin/Train/*"):
    label = directory_path.split("\\")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)



test_images = []
test_labels = [] 
for directory_path in glob.glob("D:/Ayush/Rabbin/Test/*"):
    label = directory_path.split("\\")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        # img = enhance_image(img)
        # img = remove_hair(img) 
        test_images.append(img)
        test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


#x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = (train_images, test_images, train_labels,test_labels)

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# label_encoder = LabelEncoder()

# Fit and transform the y_train labels
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.fit_transform(y_test)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)





# def AcsConv_block(dense_layer):
#     data_format = tf.keras.backend.image_data_format()
#     x3 = tf.keras.layers.Conv2D(dense_layer.shape[-1], 3)(dense_layer)
#     x1 = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(x3)
#     x1 = tf.keras.layers.Dense(dense_layer.shape[-1])(x1)
#     x1 = tf.keras.layers.Activation("relu")(x1)
#     x1 = tf.keras.layers.Dense(dense_layer.shape[-1])(x1)
#     x1 = tf.keras.layers.BatchNormalization()(x1)
#     x2 = tf.keras.layers.Conv2D(dense_layer.shape[-1], 3,strides=(1, 1),padding="same")(dense_layer)
#     x2 = tf.keras.layers.BatchNormalization()(x2)
#     multiplication_output = tf.keras.layers.multiply([x1, x2])
#     return multiplication_output



# def AcsConv_block(feature3):
#     x1 = tf.keras.layers.GlobalAveragePooling2D()(feature3)
#     x1 = tf.keras.layers.Reshape((1, 1, feature3.shape[-1]))(x1)
#     x1 = tf.keras.layers.DepthwiseConv2D(1)(x1)
#     x1 = tf.keras.layers.Activation('relu')(x1)
#     x1 = tf.keras.layers.DepthwiseConv2D(1)(x1)
#     x1 = tf.keras.layers.BatchNormalization()(x1)
#     x2 = tf.keras.layers.Conv2D(feature3.shape[-1],1)(feature3)
#     x2 = tf.keras.layers.BatchNormalization()(x2)
#     output = tf.keras.layers.multiply([x1,x2])
#     return output

import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Add, Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Lambda, Reshape, Multiply, MaxPooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# def sk_block1(input, filters, strides=1):
#     shortcut = Conv2D(4*filters, (1, 1), strides=strides)(input)
#     shortcut = BatchNormalization()(shortcut)

#     x = Conv2D(filters, (1, 1), strides=1)(input)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)

#     x = sk_block2(x, filters, strides)

#     x = Conv2D(4*filters, (1, 1), strides=1)(x)
#     x = BatchNormalization()(x)
#     x = Add()([x, shortcut])
#     x = ReLU()(x)
#     return x

# def sk_block2(input, filters, strides=1):
#     # Branch 1
#     branch1 = Conv2D(filters, (3, 3), strides=strides, padding="same")(input)
#     branch1 = BatchNormalization()(branch1)
#     branch1 = ReLU()(branch1)
#     branch1 = Conv2D(filters, (3, 3), strides=1, padding="same")(branch1)
#     branch1 = BatchNormalization()(branch1)
#     branch1 = ReLU()(branch1)

#     # Branch 2
#     branch2 = Conv2D(filters, (5, 5), strides=strides, padding="same", dilation_rate=1)(input)
#     branch2 = BatchNormalization()(branch2)
#     branch2 = ReLU()(branch2)
#     branch2 = Conv2D(filters, (5, 5), strides=1, padding="same", dilation_rate=3)(branch2)
#     branch2 = BatchNormalization()(branch2)
#     branch2 = ReLU()(branch2)

#     fused = Add()([branch1, branch2])

#     attention = GlobalAveragePooling2D()(fused)
#     x_a = Dense(filters // 16, activation='relu')(attention)
#     x_a = Dense(2 * filters, activation='sigmoid')(x_a)
#     x_a = tf.keras.layers.Flatten()(x_a)
#     x_a = Reshape((1,1,filters))(x_a)
#     a = tf.split(x_a, num_or_size_splits=2, axis=-1)
#     # a1 = Reshape((1, 1, filters))(a[0])
#     # a2 = Reshape((1, 1, filters))(a[1])

#     output = Add()([Multiply()([branch1, a[0]]), Multiply()([branch2, a[1]])])
#     return output


from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Lambda, Reshape, Multiply
import tensorflow as tf

def sk_block2(input, filters, strides=1):
    # Branch 1
    branch1 = Conv2D(filters, (3, 3), strides=strides, padding="same")(input)
    branch1 = BatchNormalization()(branch1)
    branch1 = ReLU()(branch1)
    # branch1 = Conv2D(filters, (3, 3), strides=1, padding="same")(branch1)
    # branch1 = BatchNormalization()(branch1)
    # branch1 = ReLU()(branch1)

    # Branch 2
    branch2 = Conv2D(filters, (5, 5), strides=strides, padding="same", dilation_rate=3)(input)
    branch2 = BatchNormalization()(branch2)
    branch2 = ReLU()(branch2)
    branch3 = Conv2D(filters, (5, 5), strides=1, padding="same", dilation_rate=5)(branch2)
    branch3 = BatchNormalization()(branch3)
    branch3 = ReLU()(branch3)
    # branch4 = Add()([branch2, branch3])

    fused = Add()([branch1, branch3])

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

    output = Add()([Multiply()([branch1, a1]), Multiply()([branch3, a2])])
    return output



def multiScale_feature_fusion_block(sampled_dense_output1,sampled_dense_output2,sampled_dense_output3,sampled_AcsConv_layer):
  concat_layer = tf.keras.layers.Concatenate()([sampled_dense_output1,sampled_dense_output2,sampled_dense_output3,sampled_AcsConv_layer])
  convolution_layer = tf.keras.layers.Conv2D(128,1,padding="same")(concat_layer)
  x1 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output1])
  x2 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output2])
  x3 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output3])
  x4 = tf.keras.layers.Concatenate()([convolution_layer,sampled_AcsConv_layer])
  return [x1,x2,x3,x4]

# def multiScale_feature_fusion_block(sampled_dense_output1,sampled_dense_output2,sampled_dense_output3):
#   concat_layer = tf.keras.layers.Concatenate()([sampled_dense_output1,sampled_dense_output2,sampled_dense_output3])
#   convolution_layer = tf.keras.layers.Conv2D(128,1,padding="same")(concat_layer)
#   x1 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output1])
#   x2 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output2])
#   x3 = tf.keras.layers.Concatenate()([convolution_layer,sampled_dense_output3])
#   # x4 = tf.keras.layers.Concatenate()([convolution_layer,sampled_AcsConv_layer])
#   return [x1,x2,x3]




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







checkpoint_path = "D:/Ayush/Results/Rabbin_dataset/custom_feature_Variation/Resnet/Models/full_Resnet_rabbin.keras" 

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_best_only=True, 
                                                         monitor='val_accuracy',
                                                         save_freq="epoch",
                                                         verbose=1)




input_shape = (192,192,3)
input_tensor = Input(shape = input_shape)
#base_model = Xception(weights ='imagenet',include_top = False,input_tensor= input_tensor)
# base_model = DenseNet201(weights ='imagenet',include_top = False,input_tensor= input_tensor)
# base_model = MobileNetV2(weights ='imagenet',include_top = False,input_tensor= input_tensor)
base_model = ResNet50V2(weights ='imagenet',include_top = False,input_tensor= input_tensor)


# Make all the layers in model_2_base_model trainable
base_model.trainable = False

# # Freeze all layers except for the last 10
# for layer in base_model.layers[:-10]:
#   layer.trainable = False
  
# feature1 = base_model.get_layer('block3_sepconv2').output
# feature2 = base_model.get_layer('block4_sepconv2').output
# feature3 = base_model.get_layer('block13_sepconv2').output



base_model.summary()

#for efficientnet
feature1 = base_model.get_layer('conv3_block4_1_conv').output
feature2 = base_model.get_layer('conv4_block6_1_conv').output
feature3 = base_model.get_layer('conv5_block3_3_conv').output

#for densenet
# feature1 = base_model.get_layer('pool2_conv').output
# feature2 = base_model.get_layer('pool3_conv').output
# feature3 = base_model.get_layer('pool4_conv').output


# feature1 = base_model.get_layer('block_3_expand').output
# feature2 = base_model.get_layer('block_6_expand').output
# feature3 = base_model.get_layer('block_13_expand').output

# feature1 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(feature1)

print(feature1.shape)
print(feature2.shape)
print(feature3.shape)




# output of the AcsConv block
#AcsConv_layer = AcsConv_block(feature3)

AcsConv_layer = sk_block2(feature3,64)
AcsConv_layer.shape

newfeature1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(feature1)
# newfeature2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature2)
newfeature3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature3)
newAcsConv_layer = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(AcsConv_layer)

print(newfeature1.shape)
print(feature2.shape)
print(newfeature3.shape)
print(newAcsConv_layer.shape)

multiscalce_feature_fusion_list = multiScale_feature_fusion_block(newfeature1,feature2,newfeature3,newAcsConv_layer)

# multiscalce_feature_fusion_list = multiScale_feature_fusion_block(newfeature1,feature2,newfeature3)
#applying Context attention block
cs1 = tf.keras.layers.Conv2D(128,1,padding="same")(multiscalce_feature_fusion_list[0])
cs1 = CAB_Block(cs1)
cs2 = tf.keras.layers.Conv2D(128,1,padding="same")(multiscalce_feature_fusion_list[1])
cs2 = CAB_Block(cs2)
cs3 = tf.keras.layers.Conv2D(128,1,padding="same")(multiscalce_feature_fusion_list[2])
cs3 = CAB_Block(cs3)
cs4 = tf.keras.layers.Conv2D(128,1,padding="same")(multiscalce_feature_fusion_list[3])
cs4 = CAB_Block(cs4)

print(cs1.shape)
print(cs2.shape)
print(cs3.shape)
print(cs4.shape)


# adding all the attention layers
final_output = tf.keras.layers.Add()([cs1,cs2,cs3,cs4])
final_pooled_output = tf.keras.layers.GlobalAveragePooling2D()(final_output)
# flattened_output = tf.keras.layers.Flatten()(final_pooled_output)
flattened_output = tf.keras.layers.Dense(256,activation="relu")(final_pooled_output)
final_dense_layer = tf.keras.layers.Dense(5, activation='softmax', name='output_layer')(flattened_output)

model = Model(inputs=input_tensor, outputs=final_dense_layer)

model.summary()



model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])



datagen = ImageDataGenerator(
       rotation_range=20,
       zoom_range=0.1,
       horizontal_flip=True,
       vertical_flip=True,
      )

datagen.fit(x_train)

history = model.fit(datagen.flow(x_train,
                    y_train_one_hot,
                    batch_size = 64),
                    epochs  = 50,
                    validation_data = (x_test, y_test_one_hot),
                    callbacks=[checkpoint_callback])




import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss', fontsize=35)
plt.xlabel('Epochs', fontsize=35)
plt.ylabel('Loss', fontsize=35)
plt.legend(fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
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



from tensorflow.keras.models import load_model
import pandas as pd
# Load the model with saved weights
model = load_model('D:/Ayush/Results/Rabbin_dataset/custom_feature_Variation/Resnet/Models/full_Resnet_rabbin.keras',custom_objects=custom_objects)



from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
Y_pred = model.predict(x_test)
y_pred_classes = np.argmax(Y_pred, axis =-1)
y_true = np.argmax(y_test_one_hot, axis =-1)
confusion_M1=pd.crosstab(y_true,y_pred_classes)
#fig= plt.figure(figsize=(10,5))
#ax1=plt.subplot(121)
sns.set(font_scale=3.0) #edited as suggested
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
from scipy import interp
n_classes=5
from itertools import cycle
roc_auc_scores = []
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

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
#plt.title('Extending the ROC Curve to Multi-Class')
plt.legend(loc="lower right")
plt.show()





	




 