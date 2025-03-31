"""
This module contains all the functions and classes used in the training
module and the NN module.
"""

from skimage import measure, morphology
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal


######################### FOR DATA AUGMENTATION ##############################


def expand_dataset(img, f=214):
    """
    Creates f replicas of the img dataset, with added noise. The noise consists
    in random erosion or dilation followed by a rotation

    Parameters
    ----------
    img : numpy array (N x h x w) or (h x w)
        an array of N binary images, or one binary image. The format of the
        pixels isn't important (can be 0/1 or o/255 or True/False)

    f : int, optional
        number of noisy replicas of img. The default is 214, such that
        applying this function to our 84-pictures figure dataset will give us
        around 18000 samples to introduce in the MNIST dataset of 60000 digits

    Returns
    -------
    img_exp : numpy array (f x N x h x w)
        the unsigned 8 bit int type is adopted, to save memory

    """
    rng = default_rng()  # New instance of a generator

    if img.ndim == 2:  # If there's actually only one image
        img = img[None, :, :]

    N, h, w = img.shape

    img_exp = np.empty((f, N, h, w), dtype="uint8")  # To optimize space

    for i in range(f):

        # First: dilation / erosion

        radius = normal_pos_int(rng)

        if np.round(rng.uniform()):
            img_exp[i] = dilation(img, radius)

        else:
            img_exp[i] = erosion(img, radius)
        # Note: the result of this operation is binary, so img_exp contains 1 and 0

        # Second: flip images (1/5 chance): the network should recognize
        # figures even if they are flipped

        if rng.uniform() < 0.2:
            img_exp[i] = np.flip(img_exp[i], axis=(1, 2))

        # Third: small random rotations of the imgs around the center of mass
        angle = rng.normal(scale=3)
        img_exp[i] = gray_2_binary(rotate(img_exp[i], angle), maxval=1)

    return img_exp


def normal_pos_int(rng, mean=0, sigma=2):
    """
    Draws samples from gaussian distribution, but only if positive
    """
    x = rng.normal(loc=mean, scale=sigma)
    return np.round(x) if x >= 0 else normal_pos_int(rng, mean, sigma)


def rotate(imgs, angle):
    """
    Rotate the images around the center of mass by the given angle

    Parameters
    ----------
    imgs : numpy array
        an array of N grayscale images, or one grayscale image

    angle : float


    Returns
    -------
    imgs_r : numpy array
        the array of N grayscale images, rotated. The unsigned 8 bit int type
        is adopted, to save memory

    """

    if imgs.ndim == 2:  # If there's actually only one image
        imgs = imgs[None, :, :]

    N, h, w = imgs.shape

    imgs_r = np.empty(imgs.shape, dtype="uint8")

    for i in range(N):

        center = center_of_mass(imgs[i])
        T = cv2.getRotationMatrix2D(center, angle, scale=1)

        imgs_r[i] = cv2.warpAffine(imgs[i], T, (h, w))

    return imgs_r


################################## CLASSIFIERS ###############################


class HDClassifier:
    """
    Class used to represent a heart/diamonds classifier. The second Fourier
    descriptors are extracted from the training sets at initialization.

    Attributes
    ----------
    descriptors : list
        list containing two np arrays of size M: the second Fourier descriptors
        for the heart training set and the diamonds training set

    threshold : float
        the threshold used for classification. The default is 1000, which was
        chosen by inspecting the histogram of the training set

    Methods
    -------
    plot
    classifier
    update_thresh(new_thresh)

    """

    def __init__(self, hearts, diamonds):
        """
        Initializes class attribute "descriptors", i.e. does feature
        extraction. The second Fourier descriptors are extracted from each of
        the hearts and diamonds training datasets.

        Parameters
        ----------
        hearts : numpy array
            an array of N binary heart images
        diamonds : numpy array
            an array of N binary diamond images
        """
        self.descriptors = []
        self.descriptors.append(fourier_feature_extraction(hearts, [2]))
        self.descriptors.append(fourier_feature_extraction(diamonds, [2]))
        self.threshold = 1000

    def plot(self):
        """
        Plots the second Fourier descriptors of the training sets on a
        histogram
        """

        plt.figure()
        plt.hist(
            self.descriptors,
            bins=50,
            histtype="barstacked",
            label=["hearts", "diamonds"],
        )
        plt.legend()
        plt.xlabel("|f2| : amplitude of the second Fourier descriptor")
        plt.ylabel("Number of occurences")
        plt.show()

    def classifier(self, imgs):
        """
        Classifies images: hearts or diamonds?

        Parameters
        ----------
        imgs : numpy array
            an array of N binary images, assumed either heart or diamonds

        Returns
        -------
        pred_labels : numpy array (str32)
            of size (N). It contains the predicted label of each picture,
            either 'H' or 'D'

        """

        descriptors = fourier_feature_extraction(imgs, [2])
        pred_labels_tmp = descriptors > self.threshold  # 1 is hearts, 0 is diamonds
        pred_labels = np.empty(pred_labels_tmp.shape, dtype="str_")

        pred_labels[pred_labels_tmp] = "H"
        pred_labels[np.logical_not(pred_labels_tmp)] = "D"

        return pred_labels

    def update_thresh(self, new_thresh):
        """
        Updates the value of the threshold (set to 1000 by default) that is
        used for classification
        """

        self.threshold = new_thresh


class SCClassifier:
    """
    Class used to represent a spades/clubs classifier. The 7th and 8th Fourier
    descriptors are extracted from the training sets at initialization.

    Attributes
    ----------
    descriptors : list
        list containing two np arrays of size (M x 2): the 7th and 8th Fourier
        descriptors for the spades training set and the clubs training set
    w : numpy array of size 2

    x_0 : numpy array of size 2
        a point of the linear boundary between the two classes. Chosen
        by inspecting the graph (generated by the plot method)

    x_i : numpy array of size 2
        another point of the linear boundary between the two classes. Chosen
        by inspecting the graph (generated by the plot method). The information
        is redundant, but it's more practical

    Methods
    -------
    plot
    classifier

    """

    def __init__(self, spades, clubs):

        self.descriptors = []
        self.descriptors.append(fourier_feature_extraction(spades, [7, 8]))
        self.descriptors.append(fourier_feature_extraction(clubs, [7, 8]))
        self.x_0 = np.array([200, 150])
        self.x_1 = np.array([600, 350])
        self.w = np.array([self.x_0[1] - self.x_1[1], self.x_1[0] - self.x_0[0]])

    def plot(self):
        """
        Plots the extracted features of the training sets, along with the
        hand picked linear classifier
        """

        plt.figure()
        plt.scatter(
            self.descriptors[0][:, 0], self.descriptors[0][:, 1], label="Spades"
        )
        plt.scatter(self.descriptors[1][:, 0], self.descriptors[1][:, 1], label="Clubs")
        plt.plot([200, 600], [150, 350], label="boundary")

        plt.legend()
        plt.xlabel("|f_7|")
        plt.ylabel("|f_8|")
        plt.show()

    def classifier(self, imgs):
        """
        Classifies images: spades or clubs?

        Parameters
        ----------
        imgs : numpy array
            an array of N binary images, assumed either spades or clubs

        Returns
        -------
        pred_labels : numpy array (str32)
            of size (N). It contains the predicted label of each picture,
            either 'S' or 'C'

        """

        descriptors = fourier_feature_extraction(imgs, [7, 8])
        dist = (descriptors - self.x_0) @ self.w
        pred_labels = np.empty(dist.shape, dtype="str_")
        pred_labels[dist > 0] = "S"
        pred_labels[dist < 0] = "C"

        return pred_labels


def df_NN_prediction(img, modelNN, Figures=False):
    """
    Predicts the label of digits and figures, or just digits, using the
    appropriate trained Neural Network. 'df' stands for digits-figures

    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image, corresponding to the
        segmented figures or digits

    modelNN : Sequential
        the trained neural network

    Figures : bool
        whether the model was trained on figure as well, or not. The default
        is False

    Returns
    -------
    pred_not_hot : numpy array (str32)
        the predicted labels

    """

    if Figures:
        lbl = np.array(
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "J", "Q", "K"]
        )
    else:
        lbl = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

    if img.ndim == 2:  # If there's actually only one image
        img = img[None, :, :]

    N = img.shape[0]

    img_MNIST_format = MNIST_compatible(img)

    img_vec = img_MNIST_format.reshape(N, 28 * 28) / 255

    pred = modelNN.predict(img_vec)

    # Winner takes all
    pred_w = np.zeros(pred.shape, dtype="bool")
    pred_w[range(N), pred.argmax(1)] = True

    pred_not_hot = np.zeros(N, dtype="str_")

    for i in range(N):

        pred_not_hot[i] = lbl[pred_w[i]][0]

    return pred_not_hot


class SuitsClassifier:
    def __init__(self, modelNN):
        self.modelNN = modelNN

    def classifier(self, img):
        """
        Predicts label (H, D, S, C) of images. Very repetitive with function
        df_NN_classification

        Parameters
        ----------
        img : numpy array
            an array of N binary images (the segmented suits)

        Returns
        -------
        pred_not_hot : numpy array (str32)
            of size (N). It contains the predicted label of each picture

        """

        lbl = np.array(["H", "D", "S", "C"])

        if img.ndim == 2:  # If there's actually only one image
            img = img[None, :, :]

        N, h, w = img.shape

        img_centered = gray_2_binary(centering(img, L=180)) / 255

        img_MNIST_format = MNIST_compatible(img_centered)

        img_vec = img_MNIST_format.reshape(N, 28 * 28) / 255

        pred = self.modelNN.predict(img_vec)

        # Winner takes all
        pred_w = np.zeros(pred.shape, dtype="bool")
        pred_w[range(N), pred.argmax(1)] = True

        pred_not_hot = np.zeros(N, dtype="str_")

        for i in range(N):

            pred_not_hot[i] = lbl[pred_w[i]][0]

        return pred_not_hot


class DF_Classifier:

    def __init__(self, imgs_train, lbl, modelNN, descriptors_ID=[3, 4]):

        descriptors = fourier_feature_extraction(imgs_train, descriptors_ID)

        self.descriptors_ID = descriptors_ID
        self.means, self.covs = param_estimate(descriptors, lbl)  # training
        self.modelNN = modelNN  # already trained

    def classifier(self, imgs):
        """
        Predicts label (0, 1, ..., J, Q, K) of images

        Parameters
        ----------
        imgs : numpy array
            an array of N binary images, either a figure or a digit

        Returns
        -------
        pred_labels : numpy array (str32)
            of size (N). It contains the predicted label of each picture

        """
        fig_lbls = ["J", "Q", "K"]

        descriptors = fourier_feature_extraction(imgs, self.descriptors_ID)
        N = descriptors.shape[0]
        pred_labels = np.empty(N, dtype="str_")

        for i in range(N):
            for j in range(len(fig_lbls)):
                p = multivariate_normal.pdf(
                    descriptors[i], mean=self.means[j], cov=self.covs[j]
                )

                if p > 1e-8:

                    pred_labels[i] = fig_lbls[j]
                    Next = True
                    break
                else:
                    Next = False

            if not Next:
                pred_labels[i] = df_NN_prediction(imgs[i], self.modelNN).item()

        return pred_labels


########################## FOURIER DESCRIPTORS ###############################


def fourier_feature_extraction(imgs, descriptors_ID=[1, 2]):
    """
    Performs the Fourier feature extraction.

    Parameters
    ----------
    imgs : numpy array
        an array of N binary images, or one binary image
    descriptors_ID : list, optional
        list of the indices of desired fourier descriptors. The default is [1,2].

    Returns
    -------
    descriptors : numpy array (N x number of descriptors)
        each row of the array contains the desired fourier descriptors
        (starting from f1) for the corresponding complex signal. If number of
        descriptors = 1, then descriptors is of size (N)

    """

    contours = contours_extraction(imgs)
    contours_clean = contours_cleaning(contours)
    sign = contours2signals(contours_clean)
    descriptors = np.squeeze(dft_computation(sign, descriptors_ID))
    return descriptors


def param_estimate(data, lbl, lbl_of_interest=["J", "Q", "K"]):
    """
    Estimates the parameters (mean and covariance matrix) for each class. We
    use this function on the fourier descriptors

    Parameters
    ----------
    data : numpy array (N x 2)
        N 2-dimentional feature vectors
    lbl : numpy array (N) of string
        the label of each data
    lbl_of_interest : list, optional
        the labels of interest, i.e. the L classes of which we want to estimate
        the parameters

    Returns
    -------
    means : numpy array (L x 2)
    covs : numpy array (L x 2 x 2)

    """
    L = len(lbl_of_interest)

    means = np.empty((L, 2))
    covs = np.empty((L, 2, 2))

    for c, i in zip(lbl_of_interest, range(L)):
        means[i] = np.mean(data[lbl == c], axis=0)
        covs[i] = np.cov(data[lbl == c], rowvar=False)

    return means, covs


def contours_extraction(img):
    """
    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image

    Returns
    -------
    contours : list
        a list of N elements. Each element is a list containing the extracted
        countours (=arrays) for each image

    """

    contours = []

    if img.ndim == 2:  # If there's actually only one image
        img = img[None, :, :]

    N = img.shape[0]
    for i in range(N):
        contours.append(measure.find_contours(img[i], 0))

    return contours


def contours_cleaning(contours_list):
    """
    Parameters
    ----------
    contours_list : list
        a list of N element. Each element is a list of M contours (=arrays)

    Returns
    -------
    contours_list_clean : list
        a list of N element. Each element is now an array, corresponding to
        the longest contour

    """

    contours_list_clean = []

    N = len(contours_list)

    for i in range(N):  # for each list of contours (i.e. for each image)

        M = len(contours_list[i])

        if M == 1:  # There's only one contour, we're good
            contours_list_clean.append(contours_list[i][0])
            continue

        else:  # Find longest contour
            length_max = 0
            j_max = 0

            for j in range(M):
                contour_length = contours_list[i][j].shape[0]

                if contour_length > length_max:
                    length_max = contour_length
                    j_max = j

            contours_list_clean.append(contours_list[i][j_max])

    return contours_list_clean


def contours2signals(extracted_contours):
    """
    Parameters
    ----------
    extracted_contours : list
        a list of N arrays, each array being (Mx2) (=contour)

    Returns
    -------
    signals : list
        a list of N arrays, each array being of size M and complex

    """

    signals = []
    i = 0
    for c in extracted_contours:
        signals.append(np.empty(c.shape[0], dtype=np.cdouble))
        signals[i].real = c[:, 0]
        signals[i].imag = c[:, 1]
        i = i + 1

    return signals


def dft_computation(signals, descriptors_ID=[1, 2]):
    """
    Computes the desired fourier descriptors (amplitude)

    Parameters
    ----------
    signals : list
        a list of N arrays, each array being of size M and complex (a contour
        transformed into a complex signal)

    descriptors_ID : list
        Contains the index of the desired descriptors. The default is [1,2].

    Returns
    -------
    descriptors : numpy array (N x n_descriptor)
        each row of the array contains the desired fourier descriptors
        (starting from f1) for the corresponding complex signal

    """

    N = len(signals)
    descriptors = np.zeros((N, len(descriptors_ID)))

    for i in range(N):
        descriptors[i, :] = np.absolute(np.fft.fft(signals[i])[descriptors_ID])

    return descriptors


################### PREPROCESSING FOR MIST COMPATIBILITY #####################


def MNIST_compatible(img):
    """
    Transforms the segmented digits (and figures) into a MNIST compatible
    format, by mimicking the preprocessing that was used to assemble to MNIST
    dataset.

    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image

    Returns
    -------
    img28_28 : numpy array
        the array of the N images, transformed into MNIST format (float32 data
                                                                  type is used)

    """

    boxes = find_smallest_box(img)
    segmented = segment_digits(img, boxes)
    img20_20 = resize(segmented)
    img28_28 = centering(img20_20)

    return np.round(img28_28).astype("float32")


def find_smallest_box(img):
    """
    Finds the smallest rectangle in which the foreground of a binary image fits
    In this project, we use it to "crop" the digits

    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image

    Returns
    -------
    box : list
        Each of the N elements of the list is a numpy array of type int,
        representing the box computed for the corresponding image. The
        following format is used: [x_min, x_max, y_min, y_max]

    """

    box = []
    contours = contours_extraction(img)

    if img.ndim == 2:  # If there's actually only one image
        img = img[None, :, :]

    for contour in contours:
        box_temp = np.array([500, 0, 500, 0])  # 500 is the size of the images

        for c in contour:  # c is a numpy array
            xy_max = c.max(axis=0)
            xy_min = c.min(axis=0)

            if xy_min[0] < box_temp[0]:
                box_temp[0] = xy_min[0]

            if xy_min[1] < box_temp[2]:
                box_temp[2] = xy_min[1]

            if xy_max[0] > box_temp[1]:
                box_temp[1] = xy_max[0]

            if xy_max[1] > box_temp[3]:
                box_temp[3] = xy_max[1]

        box.append(box_temp)

    return box


def segment_digits(imgs, boxes):
    """
    Segment images with the given boxes

    Parameters
    ----------
    imgs : numpy array
        an array of N binary images, or one binary image
    boxes : list
        the correspoing list of N boxes, as computed by find_smallest_box

    Returns
    -------
    segmented_digits : list
        a list of N binary images, segmented from imgs using boxes

    """
    if imgs.ndim == 2:  # If there's actually only one image
        imgs = imgs[None, :, :]

    segmented_digits = []
    for img, box in zip(imgs, boxes):
        segmented_digits.append(img[box[0] : box[1], box[2] : box[3]])

    return segmented_digits


def resize(imgs, L=20):
    """
    We use this function to resize the segmented digits to 20x20 square images
    We want to keep the aspect ratio of the original digit

    Parameters
    ----------
    imgs : list
        a list of N binary images, each with variable size, corresponding to
        the segmented digits

    L : int, optional
        Size of the squared output image. It is assumed that L is smaller than
        the smaller size of any image in imgs. The default is 20

    Returns
    -------
    resized : numpy array
        an array of N images, of size LxL. The images are now grayscale,
        because of the interpolation

    """

    N = len(imgs)

    resized = np.zeros((N, L, L))

    for img, i in zip(imgs, range(N)):

        # Determine scaling factor
        f = L / max(img.shape)

        # Rescale while preserving aspect ratio
        img_scaled = cv2.resize(
            img.astype(float) * 255, None, fx=f, fy=f, interpolation=cv2.INTER_AREA
        )

        # Adapt to L x L format with zero padding
        h, w = img_scaled.shape
        x_0, y_0 = round((L - w) / 2), round((L - h) / 2)
        resized[i, y_0 : y_0 + h, x_0 : x_0 + w] = img_scaled

    return resized


def centering(imgs, L=28):
    """
    Center the images on new L x L images by placing the center of mass at the
    center of the L x L images

    Parameters
    ----------
    imgs : numpy array
        an array of N grayscale images, or one grayscale image, that have size
        l x l
    L : int, optional
        Size of the square output images. The default is 28, to mimick MNIST

    Returns
    -------
    imgs_L : numpy array
        an array of N grayscale images

    """

    if imgs.ndim == 2:  # If there's actually only one image
        imgs = imgs[None, :, :]

    N = imgs.shape[0]  # nb of images

    M = (L - 1) / 2  # center coordinate of the new image

    imgs_L = np.empty((N, L, L))

    for i in range(N):

        c_x, c_y = center_of_mass(imgs[i])
        T = np.array([[1, 0, M - c_x], [0, 1, M - c_y]])  # Displacement matrix
        imgs_L[i] = cv2.warpAffine(imgs[i], T, (L, L))

    return imgs_L


def center_of_mass(img):
    """
    Computes the center of mass of an image

    Parameters
    ----------
    img : numpy array
        a single grayscale image

    Returns
    -------
    center : tuple
        the center of mass, which is a 2D point. Integer coordinates would
        indicate that the center of mass is at the center of a pixel (hence
        it's that pixel). In other words, a coordinate x = 0 indicates the
        center (along the x axis) of pixel 0, while x = 0.5 indicates the
        midpoint between pixel zero and one

    """
    h, w = img.shape  # height and width of the image

    x_id = np.broadcast_to(np.arange(w), (h, w))
    y_id = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))

    center_x = np.average(x_id, weights=img)
    center_y = np.average(y_id, weights=img)

    return (center_x, center_y)


def gray_2_binary(img, maxval=255):

    thresh = maxval // 2
    _, img_binary = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
    return img_binary


####################### MATHEMATICAL MORPHOLOGY ##############################


def closing(img, radius=3):
    """
    Remove small dark spots (pepper)

    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image
    radius : int, optional
        the radius of the disk structuring element

    Returns
    -------
    img_clean : numpy array
        same array of binary images, after morphological closing
    """
    if img.ndim == 2:  # If there's actually only one image
        img_clean = morphology.binary_closing(img, morphology.disk(radius))
        return img_clean

    N = img.shape[0]
    img_clean = np.zeros(img.shape)
    for i in range(N):
        img_clean[i] = morphology.binary_closing(img[i], morphology.diamond(radius))
    return img_clean


def opening(img, radius=3):
    """
    Remove small bright spots (salt)

    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image
    radius : int, optional
        the radius of the disk structuring element

    Returns
    -------
    img_clean : numpy array
        same array of binary images, after morphological closing
    """
    if img.ndim == 2:  # If there's actually only one image
        img_clean = morphology.binary_opening(img, morphology.disk(radius))
        return img_clean

    N = img.shape[0]
    img_clean = np.zeros(img.shape)
    for i in range(N):
        img_clean[i] = morphology.binary_opening(img[i], morphology.disk(radius))
    return img_clean


def erosion(img, radius=3):
    """
    The white foreground shrinks

    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image
    radius : int, optional
        the radius of the disk structuring element

    Returns
    -------
    img_eroded : numpy array
        same array of binary images, after morphological erosion
    """
    if img.ndim == 2:  # If there's actually only one image
        img_eroded = morphology.binary_erosion(img, morphology.disk(radius))
        return img_eroded

    N = img.shape[0]
    img_eroded = np.zeros(img.shape)
    for i in range(N):
        img_eroded[i] = morphology.binary_erosion(img[i], morphology.disk(radius))
    return img_eroded


def dilation(img, radius=3):
    """
    The black background shrinks

    Parameters
    ----------
    img : numpy array
        an array of N binary images, or one binary image
    radius : int, optional
        the radius of the disk structuring element

    Returns
    -------
    img_dil : numpy array
        same array of binary images, after morphological dilation
    """
    if img.ndim == 2:  # If there's actually only one image
        img_dil = morphology.binary_erosion(img, morphology.disk(radius))
        return img_dil

    N = img.shape[0]
    img_dil = np.zeros(img.shape)
    for i in range(N):
        img_dil[i] = morphology.binary_erosion(img[i], morphology.disk(radius))
    return img_dil
