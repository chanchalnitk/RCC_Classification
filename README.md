# RCC_Classification
import tensorflow.keras.backend as K 

from tensorflow.keras.models import Model
from  tensorflow.keras.initializers import *
from tensorflow.keras.layers import*
import collections
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)
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
    def se_block1(block_input, num_filters, ratio=2):

	pool1 = tf.keras.layers.GlobalAveragePooling2D()(block_input)
	flat = tf.keras.layers.Reshape((1, 1, num_filters))(pool1)
	dense1 = tf.keras.layers.Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = tf.keras.layers.Dense(num_filters, activation='sigmoid')(dense1)
	scale = tf.keras.layers.multiply([block_input, dense2])
	
	return scale
def se_block2(block_input, num_filters, ratio=4):

	pool1 = tf.keras.layers.GlobalAveragePooling2D()(block_input)
	flat = tf.keras.layers.Reshape((1, 1, num_filters))(pool1)
	dense1 = tf.keras.layers.Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = tf.keras.layers.Dense(num_filters, activation='sigmoid')(dense1)
	scale = tf.keras.layers.multiply([block_input, dense2])
	
	return scale
def se_block3(block_input, num_filters, ratio=8):

	pool1 = tf.keras.layers.GlobalAveragePooling2D()(block_input)
	flat = tf.keras.layers.Reshape((1, 1, num_filters))(pool1)
	dense1 = tf.keras.layers.Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = tf.keras.layers.Dense(num_filters, activation='sigmoid')(dense1)
	scale = tf.keras.layers.multiply([block_input, dense2])
	
	return scale

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
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ["acc"])
model.summary()
