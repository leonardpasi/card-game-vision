"""
We used this module to train Model_9527, ModelFD and ModelS.

Model_9527 is trained on the MNIST dataset, its goal is to classify digits.
Hence, it has 10 output units. The exact code we used is no longer in the
module, but since it is exactly what we did for lab3, we don't see that as an
issue. This model is the one we use in our final approach.

ModelFD's goal was to classify digits and figures at once. The code to
generate and train this model is below. First we create a dataset, which
consists in the MNIST dataset plus the figures (J, Q, K) that we extracted
from the training games. This set of figures has been augmented, in order to
artificially create a dateset of the proportions of MNIST. In other words, it
has been replicated with random noise (rotations and dilations or erosions).
Of course, it has also been processed to fit the MNIST format. As explained in
module training, this approach turned out to be a failure.

ModelS' goal is to classify the suit. The code to generate and train this
model is below. The training set consists in the segmented suits images, that
we preprocess to reduce dimentionality (MNIST format is used) and we augment.

All the models are saved in the 'NN' folder, such that there is no point in
running this module again.

"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import Input

from functions import *
import functions_bis as f

# %%

##############################################################################
############################## MODEL FD ######################################
##############################################################################


########################## EXTRACTING THE DATA ###############################

train_digits, test_digits, train_dig_labels, test_dig_labels = f.load_MNIST()

fig_lbl, fig_pic = f.load_segmented_figures()
fig_pic_exp = expand_dataset(fig_pic, fx=214)  # Replicate with noise fx = 214 times


# %% ###################### MERGING THE DATASETS ##############################

fx, N, h, w = fig_pic_exp.shape
fig_pic_exp_b = fig_pic_exp.reshape((fx * N, h, w))

train_figures = MNIST_compatible(fig_pic_exp_b)

train_fig_labels = np.tile(fig_lbl, fx)
train_fig_lbl_int = f.convert_lbl_2int(train_fig_labels)

train_data = np.concatenate((train_digits, train_figures))
train_labels = np.concatenate((train_dig_labels, train_fig_lbl_int))
# No need to shuffle, it is done automatically when training the model


# %% #################### FORMATTING THE DATASET ##############################


dim_data = np.prod(train_data.shape[1:])  # 28 * 28 = 784

# vectorize and rescale train and test data
train_data_v = train_data.reshape(train_data.shape[0], dim_data) / 255
test_data_v = test_digits.reshape(test_digits.shape[0], dim_data) / 255

# One hot encoding
train_labels_hot = to_categorical(train_labels)
test_labels_hot = to_categorical(test_dig_labels, num_classes=13)


# %% ###################### CREATING THE MODEL ################################

# define neural network
model = models.Sequential()
# input layer
model.add(Input(shape=(dim_data,)))

# first hidden layer
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# second hidden layer
model.add(Dense(36, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# output layer
model.add(Dense(13, activation="softmax"))

learning_rate = 0.001
epochs = 20
decay_rate = learning_rate / (epochs * 2)

# We use an adam optimizer with a decreasing learning rate
# loss : categorical crossentropy (well suited for multiclassification tasks)
# metric : categorical accuracy

optimizer = Adam(lr=0.1, decay=decay_rate)
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=categorical_accuracy
)


# %% ######################## TRAINING THE MODEL ##############################

model.fit(train_data_v, train_labels_hot, epochs=epochs, batch_size=16)


# %% ######################## EVALUATE THE MODE ###############################

# Test on the digits (from MNIST)

model.evaluate(test_data_v, test_labels_hot)


# Test on the original figures

test_fig_lbl_int = f.convert_lbl_2int(fig_lbl)
test_fig_lbl_hot = to_categorical(test_fig_lbl_int)

test_fig = MNIST_compatible(fig_pic)
test_fig_v = test_fig.reshape(test_fig.shape[0], dim_data) / 255

model.evaluate(test_fig_v, test_fig_lbl_hot)

# %% ######################## SAVE THE MODEL ##################################

model.save("NN/modelFD")  # FG stands for Figure-Digit

# %%

##############################################################################
############################## MODEL S #######################################
##############################################################################


data = np.load("generated_sample.npz")
suits_lbl = data["test_data_suits.npy"]
suits_pic = data["train_data_suits.npy"]

suits_pic = np.delete(suits_pic, [106, 184, 185, 214, 274], axis=0)
suits_lbl = np.delete(suits_lbl, [106, 184, 185, 214, 274])  # corrections

### LABEL CORRECTIONS:
suits_lbl[204] = "S"
suits_lbl[[88, 327]] = "C"
suits_lbl[206] = "H"
suits_lbl[321] = "D"

suits_lbl_int = f.convert_lbl_2int(suits_lbl, Figures=False)

N, h, w = suits_pic.shape
fx = 150  # expansion factor
suits_pic = gray_2_binary(centering(suits_pic, L=180))  # centering
suits_exp = expand_dataset(suits_pic, f=fx).reshape((fx * N, h, w))
suits_lbl_int = np.tile(suits_lbl_int, fx)
suits_MNIST = MNIST_compatible(suits_exp)

# vectorize
dim_data = np.prod(suits_MNIST.shape[1:])  # 28 * 28 = 784
suits_v = suits_MNIST.reshape(suits_MNIST.shape[0], dim_data) / 255

# One hot encoding
suits_lbl_hot = to_categorical(suits_lbl_int)

# %%

## Create test data
fx = 10
suits_exp_test = expand_dataset(suits_pic, f=fx).reshape((fx * N, h, w))


suits_MNIST_test = MNIST_compatible(suits_exp_test)

suits_v_test = suits_MNIST_test.reshape(suits_MNIST_test.shape[0], dim_data) / 255

suits_lbl_int_test = f.convert_lbl_2int(suits_lbl, Figures=False)
suits_lbl_int_test = np.tile(suits_lbl_int_test, fx)
suits_lbl_hot_test = to_categorical(suits_lbl_int_test)

# %% ###################### CREATING THE MODEL ################################

modelS = models.Sequential()
modelS.add(Input(shape=(dim_data,)))

# first hidden layer
modelS.add(Dense(5, activation="relu"))
modelS.add(BatchNormalization())
modelS.add(Dropout(0.3))

# second hidden layer
modelS.add(Dense(3, activation="relu"))
modelS.add(BatchNormalization())
modelS.add(Dropout(0.2))

# output layer
modelS.add(Dense(4, activation="softmax"))

learning_rate = 0.001
epochs = 3
decay_rate = learning_rate / (epochs * 2)

optimizer = Adam(lr=0.1, decay=decay_rate)
modelS.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=categorical_accuracy
)

# %% ######################## TRAINING THE MODEL ##############################

modelS.fit(suits_v, suits_lbl_hot, epochs=epochs, batch_size=16)

# %% ######################## EVALUATE THE MODE ###############################

modelS.evaluate(suits_v_test, suits_lbl_hot_test)

# %% ######################## SAVE THE MODEL ##################################

modelS.save("NN/modelS")  # FG stands for Figure-Digit
