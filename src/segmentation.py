import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage import exposure
from scipy.spatial import distance
import cv2


def get_centers_contours (sorted_contours, len_contours):
    """
    get the center coordinates of contours
    """
    centers = np.zeros((len_contours, 2))

    for i in range (0, len_contours):
        M = cv2.moments(sorted_contours[i].copy())
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers[i, 0] = cX
        centers[i, 1] = cY

    return centers

def find_suits (contours_card) :
    """
    INPUT:
    The contours list of the card
    OUTPUT:
    - The contour of the suit
    - The contour(s) of the digits
    """
    #get the total number of contours of the card
    len_contours = len(contours_card)

    #Sort the contours with the contour area
    sorted_contours = sorted(contours_card.copy(), key=lambda x: cv2.contourArea(x))

    #Get the centers of contours
    centers = get_centers_contours (sorted_contours.copy(), len_contours)
    
    #Get the contours ratios with minimum enclosing cirlce
    ratio_contours = [get_ratio_area_enclosed_area (sorted_contours[i].copy()) for i in range (len_contours)]
    
    #Since the ratios are sorted, we don't need to compute the difference matrix
    #We don't take too near (in distance) neighbors (because suits are always far one from another)
    diffs_lengthes = np.array([np.abs(ratio_contours[i+1] - ratio_contours[i])\
                             if distance.euclidean(centers[i,:], centers[i+1,:])>600 else 100 for i in range (len_contours-1) ])

    suit1 = np.argmin(diffs_lengthes)
    suit2 = suit1 + 1
    #get the highest suit (in order to have consistency about results)
    if centers[suit1, 1] > centers[suit2, 1] :
        suit_to_be_returned = suit1
    else :
        suit_to_be_returned = suit2

    y_interval = sorted([centers[suit1, 1], centers[suit2, 1]])
    x_interval = sorted([centers[suit1, 0], centers[suit2, 0]])

    def is_contained_in_suits(contour_center, x_interval=x_interval, y_interval=y_interval):
        """
        function that helps identify whether a contour is part of digits or not
        """
        return ((contour_center[0]> x_interval[0])\
                and (contour_center[0]< x_interval[1])\
                and (contour_center[1]> y_interval[0])\
                and (contour_center[1]<y_interval[1]))

    #Because there can be problems with the contours, 
    #we put one more condition (in the rectangle bounded on opposite extremities by the centers of suits)
    #to ensure that the contour is indeed a number not noise
    contour_digit = [sorted_contours[i] for i in range (len_contours) if is_contained_in_suits(centers[i,:])]
    contour_suit = [sorted_contours[suit_to_be_returned]]
    if len(contour_digit) == 0 :
        contour_digit = [sorted_contours[i] for i in range (len_contours) if ((i!=suit1) and (i!=suit2))]

    return contour_suit, contour_digit

def get_contours_one_card (img, ordered_contours_cards, num_player):
    """
    Get the concours for a given card
    """
    cimg_copy = np.zeros((img.shape[0], img.shape[1]))

    #Draw the contours on a new image
    one_card = cv2.drawContours(cimg_copy.copy(), ordered_contours_cards.copy(), num_player, color=255, thickness=-1)
    
    mask_cards_only = one_card < 1
    img_cards_only = img.copy()
    img_cards_only[mask_cards_only, : ] = 0

    #Put image to ggrescale
    greyscale_one_card = cv2.cvtColor(img_cards_only.copy(), cv2.COLOR_BGR2GRAY)

    #Apply otsu threshold method to image, rotate it so that each all players are aligned the same and get the contours
    _,img_cards_only_copy = cv2.threshold(greyscale_one_card.copy(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_cards_only_copy = np.rot90(img_cards_only_copy, k=-num_player)
    contours_card, _ = cv2.findContours(img_cards_only_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #In order to omit noise, we put 2 conditions on the contours
    contours_card = [contour_ for contour_ in contours_card if (cv2.contourArea(contour_)>800 and cv2.arcLength(contour_, closed=True)<2000)]
    contours_card = sorted(contours_card, key=lambda x: cv2.contourArea(x))[::-1]

    return contours_card


def get_needed_contours (final_contours, used_for:str):
    """
    draw the contours on a new image (to be returned)
    used_for: takes args 'digits' or 'suits': whether we want to get the digits image or the suits image
    """
    #get minimum coordinates of contours: used as reference
    min_x = min([np.min(contour_[:,0,0]) for contour_ in final_contours])
    min_y = min([np.min(contour_[:,0,1]) for contour_ in final_contours])

    numb_to_add = 1
    if used_for=='digits': numb_to_add = 100
    #draw contours on image
    for i in range (len(final_contours)) :
        final_contours[i][:,0,0] = final_contours[i][:,0,0] - min_x + numb_to_add
        final_contours[i][:,0,1] = final_contours[i][:,0,1] - min_y + numb_to_add
    #The image size is whether 500x500 or 180x180 depending whether we are looking for digits or suits
    if used_for=='digits':
        img_cards_only_copy = np.zeros((500, 500))
    else :
        img_cards_only_copy = np.zeros((180, 180))

    drawn_contour = cv2.drawContours(img_cards_only_copy.copy(), final_contours, -1, color=255, thickness=-1)
    return drawn_contour

def get_centers_and_ordered (cntsSorted ,center_badge_idx):
    """
    INPUTS:
    - cntsSorted: sorted contours
    - center_badge_idx: index of the badge

    OUTPUTS:
    - contours reordered with respect to the player number
    - the badge coordinates 
    - center coordinates reordered with respect to the player number
    """
    #Get the centers of each contour
    centers = get_centers_contours (cntsSorted, len(cntsSorted))

    #get centers of cards only as numpy array
    centers_cards = np.array([centers[i,:] for i in range (5) if i!= center_badge_idx])

    #Rearrange centers with respect to the players' names
    player1 = np.argmax(centers_cards[:,1])
    player3 = np.argmin(centers_cards[:,1])
    player2 = np.argmax(centers_cards[:,0])
    player4 = np.argmin(centers_cards[:,0])
    players_args = [player1, player2, player3, player4]

    center_badge_coordinates = centers[center_badge_idx,:]
    centers_ordered = np.array([centers_cards[i,:] for i in players_args])
    ordered_contours_cards = [cntsSorted[i] for i in players_args]
    return center_badge_coordinates, centers_ordered, ordered_contours_cards

def dilate_image(img, nb_iters=5, dilate=False):
    """
    apply a morphological change to the image
    the change is whether a dilation (with a number of terations that can be chosen) or a simple close
    """
    if dilate :
        return cv2.morphologyEx(img,cv2.MORPH_DILATE,kernel,iterations = nb_iters)
    else :
        return cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations = nb_iters)

def apply_hull (contours):
    """
    Apply convex hull to contours
    Helps when not all contours are visible
    """
    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i].copy(), False))
    return hull

def get_ratio_area_enclosed_area (closed_contour):
    """
    compute area of contour and the area of its enclosed circle and return the ratio
    Helps us find the dealer
    """
    #compute the radius of the minimum enclosing circle of the contour
    min_circle_radius = cv2.minEnclosingCircle(closed_contour)[-1]
    
    #get the area of the minimum enclosing circle of the contour
    area_enclosing = np.pi * min_circle_radius **2

    #get the area of the contour
    area = cv2.contourArea(closed_contour)

    #return the ratio of the area of the contour by the area of the minimum enclosing circle
    return area/area_enclosing


def get_important_contours (img, dilate=False, nb_iters=5, hull=False, delay_pixel=0, gamma=1.51):
    """
    1) Adjust gamma 
    2) Apply a green filter that will return a greyscale image depending whether the pixel is green or not
    3) Possibility to apply dilation (with chosen number of iterations) on the greyscale image
    4) Find the contours of the image
    5) Possibility to apply a hull on the contours
    6) sort the contours depending on the length and keep the 5 first ones
    If the contours have been well retrieved, the first four represent the cards while the fifth representts the dealer 

    INPUTS:
    - img : The original image
    - dilate: Boolean: Whether we want to dilate the image or not (after passing the green filter)
    - nb_iters: number of iterations applied in the dilation
    - hull: Boolean: whether we want to apply a hull on the contours or not
    - delay_pixel: Int: number we put if we want the threshold of green pixels to be larger

    OUTPUT:
    The 5 largest contours (representing the four cards and the dealer badge)
    """
    green_contours = get_green_contours(img.copy(), delay_pixel)

    #dilate the contours of the image
    final_im = dilate_image(green_contours.copy(), nb_iters=5, dilate=False)

    #Get contours
    contours_tmp, _ = cv2.findContours(final_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if hull : 
        contours2 = apply_hull (contours_tmp.copy())
    else :
        contours2 = contours_tmp.copy()

    cntsSorted = sorted(contours2.copy(), key=lambda x: cv2.arcLength(x, True))[::-1][:5]
    return cntsSorted

kernel = np.ones((5,5),np.uint8)
def get_green_contours (img, delay_pixel=0) :
    """
    function that return a greyscale image after applying a mask that detects green colours 
    delay_pixel: Int : number with which we want to change the green mask interval (the bigger the wider the interval)
    """
    #Convert image to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Get the mask
    mask = cv2.inRange(hsv, (40-delay_pixel, 25-delay_pixel, 25-delay_pixel), (80+delay_pixel, 255,255))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = 255
    greyscale = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    return greyscale

def get_digits(train=False):
    """
    Main function, used to find the digits and return them
    INPUT:
    - train: Bool: whether we are training (generating data to be able to test it) or that we are testing (no labels given)
    """
    X_digits = []
    X_suits = []
    pred_dealer = []
    image_paths = []        

    #Prepare the paths list for train or test
    if train:

        y_digits = []
        y_suits = []
        test_dealer = []

        games_paths = sorted(['game'+str(i) for i in range (1, 8)])
        rounds_paths = [str(i)+'.jpg' for i in range (1, 14)]
        for game_nb in range (0, len(games_paths)) :

            results_path = 'train_games/'+games_paths[game_nb] + '/game'+str(game_nb + 1)+'.csv'

            data_game_1 = pd.read_csv(results_path)
            
            
            for round_number in range (len(rounds_paths)) :
                #could have been done with for loop
                row = data_game_1.loc[round_number,:]
                y_suits.append(row.P1[1])
                y_digits.append(row.P1[0])

                y_suits.append(row.P2[1])
                y_digits.append(row.P2[0])

                y_suits.append(row.P3[1])
                y_digits.append(row.P3[0])

                y_suits.append(row.P4[1])
                y_digits.append(row.P4[0])

                test_dealer.append(row['D'])
                
                image_paths.append('train_games/'+games_paths[game_nb] + '/' + rounds_paths[round_number])
    else:
        rounds_paths = [str(i)+'.jpg' for i in range (1, 14)]
        for round_nb in range (len(rounds_paths)) :
            round = rounds_paths[round_nb]
            image_paths.append('game_test/' + round)
    
    #Begin processing
    counter=0
    for image_path in image_paths:
        img_original = cv2.imread(image_path)
        img = img_original.copy()
        img = exposure.adjust_gamma(img.copy(), gamma=1.51)
        cntsSorted = get_important_contours (img.copy(), dilate=True, nb_iters=5, hull=True)

        ratios_area_enclosed_area = np.array([get_ratio_area_enclosed_area (x.copy()) for x in cntsSorted.copy()])
        #Do two other preprocessings to the data in case the first one didn't work well

        #The dealer badge's ratio with the area of it's eenclose cirle is around 0.9
        #When all the ratios are inferior to 0.7, this means that the dealer wasn't found correctly
        # => dealer and card were merged because of dilate
        #We do another preprocessing but with less dilation iterations
        if np.all(ratios_area_enclosed_area<0.7 ) :
            #print('ratios<0.7')
            img = img_original.copy()
            img = exposure.adjust_gamma(img.copy(), gamma=1.4)
            cntsSorted = get_important_contours (img.copy(), dilate=True, hull=True, nb_iters=1, delay_pixel=1)
            ratios_area_enclosed_area = np.array([get_ratio_area_enclosed_area (x.copy()) for x in cntsSorted.copy()])

        #Manual expection also shows that the ratio of cards when everything is goo is around 0.6
        #When at least one ratio is inferior to 0.5, this means that one card wasn't well detected
        #We do the preprocessing again but with more iterations on dilations
        if np.any(ratios_area_enclosed_area<0.5) : 
            #print('one ratio < 0.5')
            img = img_original.copy()
            img = exposure.adjust_gamma(img.copy(), gamma=1.7)
            cntsSorted = get_important_contours (img.copy(), dilate=True, nb_iters=7, hull=True, delay_pixel=2)
            ratios_area_enclosed_area = np.array([get_ratio_area_enclosed_area (x.copy()) for x in cntsSorted.copy()])


        #HERE KNOW IF FAIL OR FALSE D

        center_badge_idx = np.argmax(ratios_area_enclosed_area)
        center_badge, centers_ordered, ordered_contours_cards = get_centers_and_ordered (cntsSorted ,center_badge_idx)


        centers_distances = np.array([np.linalg.norm(centers_ordered[i, :]-center_badge) for i in range (4)])
        badge_idx = np.argmin(centers_distances)

        pred_dealer.append(badge_idx + 1)

        #Iterate over the four players and get the digit and the suit for each
        for iter in range (0, 4) :
            contours_card = get_contours_one_card (img.copy(), ordered_contours_cards.copy(), iter)
            try :
                contour_suit, contour_digit = find_suits (contours_card)
                X_suit = get_needed_contours (contour_suit.copy(), 'suits')
                X_digit = get_needed_contours (contour_digit.copy(), 'digits')
                X_suits.append(X_suit)
                X_digits.append(X_digit)
                
                #If we are training, we also get the test labels

        
            #case there is an error: catch an exception
            except Exception as e:
                if train:
                    #Do not train on wrong data: skit it and print a warning                    
                    print('Warning: failed on -->', image_path)
                    y_digits.pop(counter)
                    y_suits.pop(counter)
                else:
                    #generate ranom images for test
                    X_suits.append(np.random.random((500, 500)))
                    X_digits.append(np.random.random((180, 180)))
            counter += 1

    #If we are ttraining, save the images locally in order not to run the program much times
    if train: 
        np.savez(
            'generated_sample.npz',
            test_data_dealer = test_dealer,
            predictions_dealer = pred_dealer,
            test_data_suits = y_suits,
            test_data_digits = y_digits,
            train_data_digits = X_digits,
            train_data_suits = X_suits
        )
    #If we are testing, return a dictionnary with the dealer predictions, suits images and digits images
    else:
        return dict(
            predictions_dealer = pred_dealer,
            train_data_digits = X_digits,
            train_data_suits = X_suits
        )