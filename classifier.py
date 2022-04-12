import tensorflow as tf
import pickle
import numpy as np

#hyperparameters that you can play with
######################################
EPOCHS = 2 
BATCH_SIZE = 64 
######################################

#now for the good bit, the neural net!

#load our data in from before
x_train = pickle.load(open("x_train.pickle","rb"))
y_train = pickle.load(open("y_train.pickle","rb"))

#x_train =x_train/255.0#normalise data for less wacky results
#stops the library from breaking, no idea why
x_train = np.array(x_train)
x_train = tf.keras.utils.normalize(x_train)
y_train = np.array(y_train)

#feel free to tweak this around as much as you want, it's definitely not optimal - just to get you started
def create_model() :
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size = (3,3), input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Conv2D(64, kernel_size = (3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Conv2D(128, kernel_size = (3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(128, kernel_size = (3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model

#train the model and save it
def train_model() :
    model = create_model()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.save("model")
    model.fit(x_train,y_train,batch_size = BATCH_SIZE, epochs = EPOCHS, verbose= 1, validation_split = 0.1 )

#test the model against unseen data, this is the real measure of accuracy
def evaluate_model() :
    x_test= pickle.load(open("x_test.pickle","rb"))
    y_test = pickle.load(open("y_test.pickle","rb"))
    x_test = tf.keras.utils.normalize(x_test)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    model = tf.keras.models.load_model("./model")
    model.evaluate(x_test, y_test, batch_size=128)


##################
#comment/uncomment as needed
train_model()
evaluate_model()