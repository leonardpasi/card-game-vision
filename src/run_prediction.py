"""
This is the module we run the predict the outcome of the new unseen game.
"""

# %%
import os
import numpy as np

from segmentation import get_digits
import utils as u
from training import digit_fig_classifier, suits_classifier

# %% Segmentation
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(repo_root, "data")

data = get_digits(data_path, train=False)
pred_dealer = np.asarray(data["predictions_dealer"])
suites = np.asarray(data["train_data_suits"])
digits = np.asarray(data["train_data_digits"])

# %% Classification

pred_digits = digit_fig_classifier.classifier(digits)
pred_suites = suits_classifier.classifier(suites)


# %% Printing results

predictions = u.format_pred(pred_digits, pred_suites)
pts_standard, pts_advanced = u.count_points(pred_digits, pred_suites, pred_dealer)

u.print_results(predictions, pred_dealer, pts_standard, pts_advanced)
