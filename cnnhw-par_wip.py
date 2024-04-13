# bthelmer Braden T Helmer
# cwkavana Colin WB Kavanaugh
## Stage 1: Installing dependencies and notebook gpu setup
import os
from sys import argv
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.keras.datasets import cifar10

#for RTX GPUs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


#refactored code to include the get_dataset() function
def get_dataset():
  #Loading the dataset
  #moved to get_dataset
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  ### Image normalization
  X_train = X_train / 255.0
  X_train.shape
  X_test = X_test / 255.0

  return (X_train, y_train), (X_test, y_test)
  
#Refactored code to include get_compiled_model function
def get_compiled_model(): 
  model = tf.keras.models.Sequential()

  ### Adding the first CNN Layer
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))

  ### Adding the second CNN Layer and max pool layer
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
  model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

  ### Adding the third CNN Layer
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

  ###  Adding the fourth CNN Layer and max pool layer
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
  model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

  ### Adding the Flatten layer
  model.add(tf.keras.layers.Flatten())

  ### Adding the first Dense layer
  model.add(tf.keras.layers.Dense(units=128, activation='relu'))

  ### Adding the second Dense layer (output layer)
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

  model.summary()

  ### Compiling the model
  model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam", metrics=["sparse_categorical_accuracy"])
              
  return model         
#end of get_compiled_model              
              



print(f"starting node code")

os.environ["KERAS_BACKEND"] = "tensorflow"

# Get input argument from command line.
EVALUATOR = -1
if not len(argv) > 1:
    print("Please provide input for evaluator (-1) or worker (0..)")
    exit(1)

Role = argv[1];


if Role == "-1":
  node_role = "evaluator"
elif Role.isdigit():
  node_role = "worker"
else:
  raise ValueError("Invalid node role specified")



## Stage 2: Importing dependencies for the project

os.environ['NCCL_P2P_DISABLE'] = "1"
# Commented out IPython magic to ensure Python compatibility.
#get a local copy of datasets
os.system("ln -s /mnt/beegfs/fmuelle/.keras ~/")


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## Stage 3: Dataset preprocessing
### Loading the Cifar10 dataset
#Setting class names for the dataset
if node_role == "worker":
  print(f"DOING WORKER NODE")
  CKPT_DIR = "/home/UNITY_ID/ckpt" #Your unity id here
  
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  #moved to get_dataset
  (X_train, y_train), (X_test, y_test) = get_dataset()

  ## Stage 4: Building a Convolutional neural network
  ### Defining the model

  with strategy.scope():
    #get the compiled model
    model = get_compiled_model()

  ## HOMEWORK SOLUTION
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CKPT_DIR, "ckpt_{epoch}"), save_freq="epoch")

  ### Training the model
  model.fit(X_train, y_train, epochs=15, callbacks=[checkpoint_callback])
  
  print(f"Ending node code")

elif node_role == "evaluator":
  print(f"DOING EVALUATOR NODE")
  strategy = tf.distribute.MirroredStrategy()
  CKPT_DIR = "/home/UNITY_ID/ckpt" #Your unity id goes here
  while True:
    subdirectories = os.listdir(CKPT_DIR)
    if subdirectories:
      for directory in subdirectories:
        try:
          ### Model evaluation and prediction
          model = tf.keras.models.load_model(os.path.join(CKPT_DIR, directory))
          #restored_model = tf.keras.models.load_model(os.path.join(CKPT_DIR, directory))
          test_loss, test_accuracy = model.evaluate(X_test, y_test)
          
          print("Test accuracy: {}".format(test_accuracy))
        except Exception as e:
          print(f"Error: {e}")
        time.sleep(1)
        
    else:
      print(".", end="", flush=True)
      time.sleep(1)

  
  
