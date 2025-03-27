import numpy as np

def format_pred(pred_digits, pred_suites):
    """
    Formats our predictions to the format required by other utils functions,
    such as evaluate_game and print_result

    Parameters
    ----------
    pred_digits : numpy array, of type str
        this one dimentional array must have a size that is a multiple of 4.
        Typically, it would be 52
        
    pred_suites : numpy array, of type str
        same size as pred_digits

    Returns
    -------
    pred : numpy array, of type str
        size (nb or rounds x 4)

    """
    
    if pred_digits.shape != pred_suites.shape:
        raise Exception("Predictions of digits and suites should have same size.")
        
    if pred_digits.shape[0] % 4 != 0:
        raise Exception("The number of predictions is not a multiple of 4")
        
    N = pred_digits.shape[0] // 4 # Number of rounds, normally 13
    
    temp = np.core.defchararray.add(pred_digits, pred_suites)
    pred = temp.reshape(N,4)
    
    return pred

def count_points(pred_digits, pred_suites, pred_dealers = None):
    """
    Counts the points of each player, in standard and (eventually) advanced
    mode

    Parameters
    ----------
    pred_digits : numpy array, of type str
        this one dimentional array must have a size that is a multiple of 4.
        Typically, it would be 52
    pred_suites : numpy array, of type str
        same size as pred_digits
    pred_dealers : numpy array of int, optional
        If this is given, then points in advanced mode are also computed. The
        default is None.

    Returns
    -------
    points_standard : numpy array
        of size 4, type int
    points_advanced : numpy array
        of size 4, type int. Returned only if pred_dealers is not None
    
    """
    points_standard = np.zeros(4, dtype=int)      ### Standard mode
    
    id_figures = np.logical_or(pred_digits == 'J', pred_digits == 'Q')
    id_figures = np.logical_or(id_figures, pred_digits == 'K')
    
    pred_dig_copy = pred_digits.copy()
    pred_dig_copy[id_figures] = '1' # Just temporary
    
    pred_dig_int = pred_dig_copy.astype(int)
    
    fig_lbls = ['J', 'Q', 'K']
    int_lbls = [10, 11, 12]

    for f_lbl, i_lbl in zip(fig_lbls, int_lbls):
        pred_dig_int[pred_digits == f_lbl] = i_lbl
    
    if pred_digits.shape != pred_suites.shape:
        raise Exception("Predictions of digits and suites should have same size.")
        
    if pred_digits.shape[0] % 4 != 0:
        raise Exception("The number of predictions is not a multiple of 4")
        
    
    for i in range(0, pred_digits.shape[0], 4): # For each round
        w = np.argwhere(pred_dig_int[i:i+4]==np.amax(pred_dig_int[i:i+4]))
        points_standard[w] += 1
        
    if pred_dealers is not None:                    ### Advanced mode
        points_advanced = np.zeros(4, dtype=int)
        
        if pred_digits.shape[0] / 4 != pred_dealers.shape[0]:
            raise Exception("Mismatch between nb of rounds and nb of predictions of the dealer")
        
        j=0
        for i in range(0, pred_digits.shape[0], 4):
            d_suite = pred_suites[i+pred_dealers[j]-1] # Suite of the dealer
            
            idx_suite = np.flatnonzero(pred_suites[i:i+4] == d_suite)
            
            w = np.argmax(pred_dig_int[i:i+4][pred_suites[i:i+4] == d_suite])
            points_advanced[idx_suite[w]] += 1
            j += 1
        
        return points_standard, points_advanced
    
    return points_standard
        
    
    
    
    


def evaluate_game(pred, cgt, mode_advanced=False):
    """
    Evalutes the accuracy of your predictions. The same function will be used to assess the 
    performance of your model on the final test game.


    Parameters
    ----------
    pred: array of string of shape NxD
        Prediction of the game. N is the number of round (13) and D the number of players (4). Each row 
        is composed of D string. Each string can is composed of 2 charcters [0-9, J, Q, K] + [C, D, H, S].
        If the mode_advanced is False only the rank is evaluated. Otherwise, both rank and colours are 
        evaluated (suits).
    cgt: array of string of shape NxD
        Ground truth of the game. Same format as the prediciton.
    mode_advanced: bool, optional
        Choose the evaluation mode
        
    Returns
    -------
    accuracy: float
        Accuracy of the prediciton wrt the ground truth. Number of correct entries divided by 
        the total number of entries.
    """
    if pred.shape != cgt.shape:
        raise Exception("Prediction and ground truth should have same shape.")
    
    if mode_advanced:
        # Full performance of the system. Cards ranks and colours.
        return (pred == cgt).mean()
    else:
        # Simple evaluation based on cards ranks only
        cgt_simple = np.array([v[0] for v in cgt.flatten()]).reshape(cgt.shape)
        pred_simple = np.array([v[0] for v in pred.flatten()]).reshape(pred.shape)
        return (pred_simple == cgt_simple).mean()
    
    
def print_results(rank_colour, dealer, pts_standard, pts_advanced):
    """
    Print the results for the final evaluation. You NEED to use this function when presenting the results on the 
    final exam day.
    
    Parameters
    ----------
    rank_colour: array of string of shape NxD
        Prediction of the game. N is the number of round (13) and D the number of players (4). Each row 
        is composed of D string. Each string can is composed of 2 charcters [0-9, J, Q, K] + [C, D, H, S].
    dealer: list of int
        Id ot the players that were selected as dealer ofr each round.
    pts_standard: list of int of length 4
        Number of points won bay each player along the game with standard rules.
    pts_advanced: list of int of length 4
        Number of points won bay each player along the game with advanced rules.
    """
    print('The cards played were:')
    print(pp_2darray(rank_colour))
    print('Players designated as dealer: {}'.format(dealer))
    print('Players points (standard): {}'.format(pts_standard))
    print('Players points (advanced): {}'.format(pts_advanced))
    
    
def pp_2darray(arr):
    """
    Pretty print array
    """
    str_arr = "[\n"
    for row in range(arr.shape[0]):
        str_arr += '[{}], \n'.format(', '.join(["'{}'".format(f) for f in arr[row]]))
    str_arr += "]"
    return str_arr
