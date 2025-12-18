#noise_range_map = {1:[0.0, 0.3], 2:[0.0, 0.6], 3:[0.0, 0.9], 4:[0.3, 1.2], 5:[0.3, 1.5], 
#                   6:[0.6, 1.8], 7:[0.6, 2.0], 8:[0.9, 2.0], 9:[1.0, 2.0], 10:[0.0, 2.0]} 

noise_range_map = {1:[0, 0.3], 2:[0.0, 0.6], 3:[0.0, 0.9], 4:[0.3, 1.2], 5:[0.3, 1.5], 
                   6:[0.6, 1.8], 7:[0.6, 2.0], 8:[0.9, 2.0], 9:[1.0, 2.0], 10:[0.6, 2.0]} 

def low_max_snr(epoch, snr_map):
    '''
    input:
    1. epoch: current epoch
    2. snr_map: a pre-defined snr map that maps from given epoch to a snr range (characterized by low and high snrs)
    return:
    1.[returned low snr, returned high snr] the low and high snrs for snr ranges in CL.
    '''

#     if epoch <= 4:
#         indicator = 1 
#     elif 4 < epoch <= 6:
#         indicator = 2
#     elif 6 < epoch <= 8:
#         indicator = 3
#     elif 8 < epoch <= 10:
#         indicator = 4
#     elif 10 < epoch <= 12:
#         indicator = 5
#     elif 12 < epoch <= 14:
#         indicator = 6
#     elif 14 < epoch <= 16:
#         indicator = 7
#     elif 16 < epoch <= 18:
#         indicator = 8
#     elif 18 < epoch <= 20:
#         indicator = 9
#     else:
#         indicator = 9   
#     return snr_map[indicator]
    a1 = 2
    a2 = a1 + 2
    a3 = a2 + 2
    a4 = a3 + 2
    a5 = a4 + 2
    a6 = a5 + 2
    a7 = a6 + 2
    a8 = a7 + 2
    a9 = a8 + 12
    
    if epoch <= a1:
        indicator = 1 
    elif a1 < epoch <= a2:
        indicator = 2
    elif a2 < epoch <= a3:
        indicator = 3
    elif a3 < epoch <= a4:
        indicator = 4
    elif a4 < epoch <= a5:
        indicator = 5
    elif a5 < epoch <= a6:
        indicator = 6
    elif a6 < epoch <= a7:
        indicator = 7
    elif a7 < epoch <= a8:
        indicator = 8
    elif a8 < epoch <= a9:
        indicator = 9
    else:
        indicator = 10 
    return snr_map[indicator]
    