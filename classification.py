"""
This is the module we will run for the examination.
"""
import matplotlib.pyplot as plt
import numpy as np

import segmentation
import utils as u
from training import digit_fig_classifier, suits_classifier

#%% Segmentation

data = segmentation.get_digits()
pred_dealer = np.asarray(data['predictions_dealer'])
suites = np.asarray(data['train_data_suits'])
digits = np.asarray(data['train_data_digits'])

#%% Classification

pred_digits = digit_fig_classifier.classifier(digits)
pred_suites = suits_classifier.classifier(suites)



#%% Printing results

predictions = u.format_pred(pred_digits, pred_suites)
pts_standard, pts_advanced = u.count_points(pred_digits, pred_suites, pred_dealer)

u.print_results(predictions, pred_dealer, pts_standard, pts_advanced)

    






