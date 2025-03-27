"""
We used this module to investigate for the best features and perform training.
The output from the segmentation module, applied to the set of training games,
is loaded. Some corrections are made on the labels. To know more about our
process, read through this module and run it with ILLUSTRATE set to True. The
boolean is set to False by default, such that when the module is ran from
the classification module, only our final approach is put into practice.
The classification module will import the classifiers that are trained in this
module. To summarize our approach:
    
    - at first, we decided to use Fourier descriptors to describe the suits,
    as well as the color black/red. However, we didn't have the time to
    extract the color, so decided to use a Neural Network, trained on an
    augmented version of the set of segmented suits from the training games
    
    - for the figures/digits, the first naive idea of using only fourier
    descriptors doesn't seem doable
    
    - the second idea consisted in having a single neural network for both
    figures and digits. The neural netword (modelFD) was trained on the MNIST
    dataset and on the segmented training figures set, which was augmented.
    Although we mimicked the preprocessing of the MNIST data set, achieving a
    format that looked almost identical to the MNIST, the NN must have learned
    to pick up on that very small difference in preprocessing. In fact, it has
    a very good performance on the MNIST testing set (>0.95) and on the figures
    (=1), but a very bad performance on the segmented digits, which are confused
    with figures.
    
    - our final approach is to combine the Fourier descriptors, used to identify
    the figures, and a Neural Network (model_9527) trained on the MNIST dataset
    to identify the digits. A new image to classify is first tested with its
    Fourier descriptors, and then eventually it's fed to the NN
"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, utils
from functions import *
import os


os.environ['KMP_DUPLICATE_LIB_OK']='True' # I need to do it on my laptop
                                          # otherwise tensorflow will crash

ILLUSTRATE = False

# Loading output from segmentation module
data = np.load('generated_sample.npz')

digits_lbl = data["test_data_digits.npy"]
digits_pic = data["train_data_digits.npy"]

suits_lbl = data["test_data_suits.npy"]
suits_pic = data["train_data_suits.npy"]

suits_pic = np.delete(suits_pic, [106, 184, 185, 214, 274], axis=0)
suits_lbl = np.delete(suits_lbl, [106, 184, 185, 214, 274]) # corrections


model_D = models.load_model('NN/model_9527') # Loading trained Neural Networks
model_S = models.load_model('NN/modelS')

if ILLUSTRATE:
    model_FD = models.load_model('NN/modelFD')


#%%
###################### TRAINING FOR SUIT CLASSIFICATION ######################

### FIRST IDEA: having the colour red/black, use Fourier descriptors to solve
### binary classification problems. Dropped because of lack of time to exctract
### the colour.

if ILLUSTRATE:
    
    # Label corrections:
    suits_lbl[204] = "S"
    suits_lbl[[88, 327]] = "C"
    suits_lbl[206] = "H"
    suits_lbl[321] = "D"


    hearts_diamonds = HDClassifier(suits_pic[suits_lbl == "H"],
                                   suits_pic[suits_lbl == "D"])
    
    spades_clubs = SCClassifier(suits_pic[suits_lbl == "S"],
                                suits_pic[suits_lbl == "C"])


    hearts_diamonds.plot()
    spades_clubs.plot()


### SECOND IDEA: do data augmentation on the extracted figures, and train a 
### neural network

suits_classifier = SuitsClassifier(model_S)

#%%

############# TRAINING FOR DIGITS AND FIGURES CLASSIFICATION #################

L = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'J', 'Q', 'K']

### LABEL CORRECTIONS:
digits_lbl[209] = 'J'
digits_lbl[332] = 'Q'
digits_lbl[207] = 'None' # Maybe it's actually a 2 that wasn't rotated?

id_figures = np.logical_or(np.logical_or(digits_lbl == 'J', digits_lbl == 'Q'),
                           digits_lbl == 'K')
figures_only_pic = digits_pic[id_figures]
figures_only_lbl = digits_lbl[id_figures]

id_digits = np.logical_not(id_figures)
digits_only_lbl = digits_lbl[id_digits]
digits_only_pic = digits_pic[id_digits]

digits_only_lbl[159] = '9'
digits_only_lbl[213] = '9'

#%%  ### FIRST IDEA: using Fourier Descriptors

if ILLUSTRATE:

    descriptors = fourier_feature_extraction(digits_pic, [3, 4])
    plt.figure()
    
    for lbl in L:
        if lbl == 'J' or lbl == 'Q' or lbl == 'K':
            m = 'x'
        else:
            m = '.'
        
        plt.scatter(descriptors[digits_lbl == lbl][:,0],
                    descriptors[digits_lbl == lbl][:,1], label = lbl, marker = m)
    plt.legend()
    plt.xlabel("|f_3|")
    plt.ylabel("|f_4|")


#%% To identify outliers

# descriptors = fourier_feature_extraction(suits_pic, [2])
# I = []
# for i in range(358):
#     if suits_lbl[i] == 'H' and descriptors[i] < 500:
#         I.append(i)
        
# print(I)

#%% ### SECOND IDEA: using Neural Network for both digits and figures

if ILLUSTRATE:


    # Testing our trained neural network on the digits
    pred = df_NN_prediction(digits_only_pic, model_FD, Figures = True)
    
    print("\nUsing an NN trained on both the MNIST and the augumented set of figures\
     from the training games, the error rate on the digits from the training games\
     is: {:.2f}".format((pred == digits_only_lbl).mean()*100))
     
     ## Very bad performance!! Even though the performance on the figures is
     ## excellent, as well as that for the MNIST testing set (100% and 95.56%
     ## respectively)




#%%  ### THIRD IDEA: combine a Neural Network trained on MNIST with Fourier
     ### descriptors to identify figures
     
digit_fig_classifier = DF_Classifier(digits_pic, digits_lbl, model_D)


     









