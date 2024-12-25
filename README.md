# Data Availability

https://drive.google.com/drive/folders/1Lz7THwUxFQVOlwKx4lfGA0l_dtqC4HkB?usp=sharing


# Important Libraries
import tensorflow.keras.backend as K 

from tensorflow.keras.models import Model
from  tensorflow.keras.initializers import *
from tensorflow.keras.layers import*
import collections
import tensorflow as tf
import tensorflow_addons as tfa
# GPU Utilization Code
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)
# SPP Block
def SPP_variant3(x1,x2,x3):
    x11 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', strides = (4,4), kernel_initializer = 'random_normal')(x1)
    x22 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', strides = (2,2), kernel_initializer = 'random_normal')(x2)
    x33 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(x3)
    x = Concatenate()([x11,x22,x33])
    x = SeparableConv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    d = GlobalAveragePooling2D()(x)
    x_out = Multiply()([x3,d])
    x_out = Conv2D(64,(1,1), activation='relu')(x_out)
    x_o = MaxPooling2D((2,2))(x_out)
    x1 = tf.keras.layers.DepthwiseConv2D((3, 3), dilation_rate = (1,1), activation='relu', padding='same')(x_o) 
    x2 = tf.keras.layers.DepthwiseConv2D((3, 3), dilation_rate = (2,2), activation='relu', padding='same')(x_o)
    x3 = tf.keras.layers.DepthwiseConv2D((3, 3), dilation_rate = (4,4), activation='relu', padding='same')(x_o)
    x4 = tf.keras.layers.DepthwiseConv2D((3, 3), dilation_rate = (6,6), activation='relu', padding='same')(x_o)
    x4 = Add()([x1,x2,x3,x4])
    x_o = Conv2D(128,(1,1), activation='relu')(x4)
    return x_o
# SE Block-1
def se_block1(block_input, num_filters, ratio=2):

	pool1 = tf.keras.layers.GlobalAveragePooling2D()(block_input)
	flat = tf.keras.layers.Reshape((1, 1, num_filters))(pool1)
	dense1 = tf.keras.layers.Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = tf.keras.layers.Dense(num_filters, activation='sigmoid')(dense1)
	scale = tf.keras.layers.multiply([block_input, dense2])
	
	return scale
# SE Block-2
def se_block2(block_input, num_filters, ratio=4):

	pool1 = tf.keras.layers.GlobalAveragePooling2D()(block_input)
	flat = tf.keras.layers.Reshape((1, 1, num_filters))(pool1)
	dense1 = tf.keras.layers.Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = tf.keras.layers.Dense(num_filters, activation='sigmoid')(dense1)
	scale = tf.keras.layers.multiply([block_input, dense2])
	
	return scale
# SE Block-3
def se_block3(block_input, num_filters, ratio=8):

	pool1 = tf.keras.layers.GlobalAveragePooling2D()(block_input)
	flat = tf.keras.layers.Reshape((1, 1, num_filters))(pool1)
	dense1 = tf.keras.layers.Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = tf.keras.layers.Dense(num_filters, activation='sigmoid')(dense1)
	scale = tf.keras.layers.multiply([block_input, dense2])
	
	return scale
# ResNet Block
def resnet_block(block_input, num_filters):

	if tf.keras.backend.int_shape(block_input)[3] != num_filters:
		block_input = tf.keras.layers.Conv2D(num_filters, kernel_size=(1, 1))(block_input)
	conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
	norm1 = tf.keras.layers.BatchNormalization()(conv1)  
	relu1 = tf.keras.layers.Activation('relu')(norm1)
	se1 = se_block1(relu1, num_filters=num_filters)
	conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
	norm2 = tf.keras.layers.BatchNormalization()(conv2)  
	relu2 = tf.keras.layers.Activation('relu')(norm2)
	se2 = se_block2(relu2, num_filters=num_filters)
	conv3 = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
	norm3 = tf.keras.layers.BatchNormalization()(conv3)  
	relu3 = tf.keras.layers.Activation('relu')(norm3)
	se3 = se_block3(relu3, num_filters=num_filters)    
	sum = tf.keras.layers.Add()([block_input, se1,se2,se3])
	relu4 = tf.keras.layers.Activation('relu')(sum)
	return relu4 
# Proposed Model
def ProposedNet(input_shape=(224,224,3), n_classes=4):
  init = Input(input_shape)
 
  x = Conv2D(16, (7, 7), activation=None, padding='same')(init) 
  x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),gamma_initializer=Constant(1.0),momentum=0.5)(x) 
  x = Activation('relu')(x)
    
#   x1 = resnet_block(x,16)
  x1 = resnet_block(x,16)
  x11 = MaxPooling2D((2,2))(x)
 
  x = Conv2D(32, (3, 3), activation=None, padding='same')(x11)
  x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),gamma_initializer=Constant(1.0),momentum=0.5)(x) 
  x = Activation('relu')(x)
  
  x2 = resnet_block(x,32)
#   x2 = resnet_block(x,32)
  x22 = MaxPooling2D((2,2))(x)

  x = Conv2D(64, (3, 3), activation=None, padding='same')(x22)
  x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),gamma_initializer=Constant(1.0),momentum=0.5)(x) 
  x = Activation('relu')(x)
  
  x3 = resnet_block(x,64)
#   x3 = MaxPooling2D((2,2))(x)

  x = SPP_variant3(x1,x2,x3) 
  avg = GlobalAveragePooling2D()(x)

  y = Dense(n_classes, activation="softmax", name="ProposedNet")(avg)
  model = Model(init, y)
  return model
model = ProposedNet(input_shape=(224,224,3), n_classes=4)
# Data Reading
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_path = 'E:/TGCA Extended Dataset-2/Training'
test_path = 'E:/TGCA Extended Dataset-2/Test'
val_path = 'E:/TGCA Extended Dataset-2/Validation'
batch_size = 4
img_height = 224
img_width = 224
no_of_classes = 4
classes_name = [0,1,2,3]
input_shape = (img_height , img_width , 3)


random_seed = np.random.seed(1142)

datagen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=True,
    horizontal_flip = False,
    vertical_flip = False,
    #validation_split = 0.1,
    featurewise_std_normalization=True)

train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = True,
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = True,
    class_mode='categorical')
print(train_generator.class_indices)
# Model Compile
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ["acc"])
model.summary()
# Model Training
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_acc' , mode='max' ,
                                                  factor = 0.5 , patience = 5 , verbose=1 , cooldown = 1,
                                                 min_delta = 0.0001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=40, verbose=1,
                                              mode = 'max', restore_best_weights = True)
check_path = 'E:/save-2/R1.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')


history_3A3 = model.fit_generator(train_generator , validation_data = validation_generator ,
                                  steps_per_epoch= len(train_generator) ,
                                  validation_steps = len(validation_generator)
                                  ,epochs = 65,callbacks = [reduce_lr, early_stop, checkpoint] )
# Learning Curve
import matplotlib.pyplot as plt
plt.plot(history_3A3['loss'] , label = 'train_loss')
plt.plot(history_3A3['val_loss'] , label = 'val_loss')
plt.legend()
plt.xlabel("No. of epochs")
plt.ylabel("Loss(Categorical Crossentropy)")
plt.title("Loss vs Epoch")
plt.savefig('E:/save-2/R1_loss.png')
plt.show()
#plt.savefig('90.98_loss.png')
plt.plot(history_3A3['acc'] , label = 'train_acc')
plt.plot(history_3A3['val_acc'] , label = 'val_acc')
plt.legend()
plt.xlabel("No. of epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch ")
plt.savefig('E:/save-2/R1_accuracy.png')
plt.show()
# Test Data
test_d = ImageDataGenerator(rescale=1. / 255)
test = test_d.flow_from_directory(
    'E:/TGCA Extended Dataset-2/Test',
    target_size=(224,224),
    batch_size=1,
    shuffle = False,
    class_mode='categorical')
import numpy as np
test_step = test.n//test.batch_size
test.reset()
pred = model.predict_generator(test , steps = test_step , verbose = 1)
pred_class_indices = np.argmax(pred,axis=1)

## printing predicted labels
print(pred_class_indices)
# Performance Metrices
from sklearn.metrics import accuracy_score,roc_curve, confusion_matrix, roc_auc_score, auc, f1_score,jaccard_score,classification_report
from sklearn.metrics import precision_score,recall_score,jaccard_score
classes = [0,1,2,3]


for cl in classes:

    print("class: ",cl)

    a1 = np.uint8(test.labels == cl)
    a2 = np.uint8(pred_class_indices == cl)

    print('Accuracy {}'.format(accuracy_score(y_true=a1, y_pred=a2)))
    print('F1 {}'.format(f1_score(y_true=a1, y_pred=a2)))
    print('precision {}'.format(precision_score(y_true=a1, y_pred=a2)))
    print('recall {}'.format(recall_score(y_true=a1, y_pred=a2)))

    print('jaccard {}'.format(jaccard_score(y_true=a1, y_pred=a2)))
    print("_______________________________")
print('Accuracy {}'.format(accuracy_score(y_true=test.labels, y_pred=pred_class_indices)))
print('F1 {}'.format(f1_score(y_true=test.labels, y_pred=pred_class_indices,average = "macro")))
print('precision {}'.format(precision_score(y_true=test.labels, y_pred=pred_class_indices,average = "macro")))
print('recall {}'.format(recall_score(y_true=test.labels, y_pred=pred_class_indices,average = "macro")))

print('jaccard {}'.format(jaccard_score(y_true=test.labels, y_pred=pred_class_indices,average = "macro")))
print('confusion_matrix\n {}'.format(confusion_matrix(y_true=test.labels, y_pred=pred_class_indices)))
print('classification_report\n {}'.format(classification_report(y_true=test.labels, y_pred=pred_class_indices)))
print('\n\n')
