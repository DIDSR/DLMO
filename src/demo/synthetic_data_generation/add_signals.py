import numpy as np

def AddSignalRayleigh(obj, A, wid, doublet_L, loc, half_size, tot_size, dim1, dim2):
# Add Rayleigh signals (binary and line) to the input object.
#
# Args:
#     obj (numpy.ndarray): Input object to add signals to.
#     A (float): Amplitude of the signal.
#     wid (float): Width of the Gaussian signal.
#     doublet_L (list): List of possible distances between dots in binary signals.
#     loc (numpy.ndarray): Array of possible locations for signals.
#     half_size (int): Half of the total number of objects.
#     tot_size (int): Total number of objects.
#     dim1, dim2 (int): Dimensions of each object.
#
# Returns:
#     tuple: (loc_list, L_index_list, obj)
#         loc_list (numpy.ndarray): List of signal locations.
#         L_index_list (numpy.ndarray): List of L indices.
#         obj (numpy.ndarray): Object with added signals.

    np.random.seed(42)
    X, Y = np.meshgrid(range(dim2), range(dim1))

    half_size = int(half_size)
    tot_size = int(tot_size)
    L_index_list = np.zeros([tot_size, 1])
    loc_list = np.zeros([tot_size, 2])

    line_dict = {'0.1': {'4': {'L': 6, 'A': 0.11}, '5': {'L': 7, 'A': 0.11}, '6': {'L': 9, 'A': 0.09}, '7': {'L': 10, 'A': 0.08}, '8': {'L': 11, 'A': 0.07}, '9': {'L': 13, 'A': 0.06}, '10': {'L': 14, 'A': 0.05}, '11': {'L': 15, 'A': 0.05}, '12': {'L': 16, 'A': 0.05}, '13': {'L': 17, 'A': 0.04}, '14': {'L': 18, 'A': 0.04}}, \
                 '0.2': {'4': {'L': 6, 'A': 0.23}, '5': {'L': 7, 'A': 0.21}, '6': {'L': 9, 'A': 0.17}, '7': {'L': 10, 'A': 0.15}, '8': {'L': 12, 'A': 0.13}, '9': {'L': 13, 'A': 0.12}, '10': {'L': 14, 'A': 0.11}, '11': {'L': 15, 'A': 0.1}, '12': {'L': 16, 'A': 0.09}, '13': {'L': 17, 'A': 0.09}, '14': {'L': 18, 'A': 0.08}}, \
                 '0.3': {'4': {'L': 6, 'A': 0.34}, '5': {'L': 7, 'A': 0.32}, '6': {'L': 9, 'A': 0.26}, '7': {'L': 10, 'A': 0.23}, '8': {'L': 12, 'A': 0.2}, '9': {'L': 13, 'A': 0.18}, '10': {'L': 14, 'A': 0.16}, '11': {'L': 15, 'A': 0.15}, '12': {'L': 16, 'A': 0.14}, '13': {'L': 17, 'A': 0.13}, '14': {'L': 18, 'A': 0.12}}, \
                 '0.4': {'4': {'L': 6, 'A': 0.46}, '5': {'L': 7, 'A': 0.42}, '6': {'L': 9, 'A': 0.35}, '7': {'L': 10, 'A': 0.31}, '8': {'L': 12, 'A': 0.26}, '9': {'L': 13, 'A': 0.24}, '10': {'L': 14, 'A': 0.22}, '11': {'L': 15, 'A': 0.2}, '12': {'L': 16, 'A': 0.19}, '13': {'L': 17, 'A': 0.18}, '14': {'L': 18, 'A': 0.16}}, \
                 '0.5': {'4': {'L': 6, 'A': 0.57}, '5': {'L': 7, 'A': 0.53}, '6': {'L': 9, 'A': 0.44}, '7': {'L': 10, 'A': 0.39}, '8': {'L': 12, 'A': 0.33}, '9': {'L': 13, 'A': 0.3}, '10': {'L': 14, 'A': 0.27}, '11': {'L': 15, 'A': 0.25}, '12': {'L': 16, 'A': 0.23}, '13': {'L': 17, 'A': 0.22}, '14': {'L': 18, 'A': 0.21}}, \
                 '0.6': {'4': {'L': 6, 'A': 0.69}, '5': {'L': 7, 'A': 0.63}, '6': {'L': 9, 'A': 0.52}, '7': {'L': 10, 'A': 0.46}, '8': {'L': 12, 'A': 0.39}, '9': {'L': 13, 'A': 0.36}, '10': {'L': 14, 'A': 0.33}, '11': {'L': 15, 'A': 0.3}, '12': {'L': 16, 'A': 0.28}, '13': {'L': 17, 'A': 0.26}, '14': {'L': 18, 'A': 0.25}}, \
                 '0.7': {'4': {'L': 6, 'A': 0.8}, '5': {'L': 7, 'A': 0.74}, '6': {'L': 9, 'A': 0.61}, '7': {'L': 10, 'A': 0.54}, '8': {'L': 12, 'A': 0.46}, '9': {'L': 13, 'A': 0.42}, '10': {'L': 14, 'A': 0.38}, '11': {'L': 15, 'A': 0.35}, '12': {'L': 16, 'A': 0.33}, '13': {'L': 17, 'A': 0.31}, '14': {'L': 18, 'A': 0.29}}, \
                 '0.8': {'4': {'L': 6, 'A': 0.91}, '5': {'L': 7, 'A': 0.85}, '6': {'L': 9, 'A': 0.7}, '7': {'L': 10, 'A': 0.62}, '8': {'L': 12, 'A': 0.52}, '9': {'L': 13, 'A': 0.48}, '10': {'L': 14, 'A': 0.44}, '11': {'L': 15, 'A': 0.4}, '12': {'L': 16, 'A': 0.38}, '13': {'L': 17, 'A': 0.35}, '14': {'L': 18, 'A': 0.33}}, \
                 '0.9': {'4': {'L': 6, 'A': 0.99}, '5': {'L': 7, 'A': 0.95}, '6': {'L': 9, 'A': 0.79}, '7': {'L': 10, 'A': 0.69}, '8': {'L': 12, 'A': 0.59}, '9': {'L': 13, 'A': 0.54}, '10': {'L': 14, 'A': 0.49}, '11': {'L': 15, 'A': 0.45}, '12': {'L': 16, 'A': 0.42}, '13': {'L': 17, 'A': 0.39}, '14': {'L': 18, 'A': 0.37}}, \
                 '1.0': {'4': {'L': 6, 'A': 1.14}, '5': {'L': 7, 'A': 1.06}, '6': {'L': 9, 'A': 0.87}, '7': {'L': 10, 'A': 0.77}, '8': {'L': 12, 'A': 0.66}, '9': {'L': 13, 'A': 0.6}, '10': {'L': 14, 'A': 0.55}, '11': {'L': 15, 'A': 0.51}, '12': {'L': 16, 'A': 0.47}, '13': {'L': 17, 'A': 0.44}, '14': {'L': 18, 'A': 0.41}}, \
                 '1.1': {'4': {'L': 6, 'A': 1.26}, '5': {'L': 7, 'A': 1.16}, '6': {'L': 9, 'A': 0.96}, '7': {'L': 10, 'A': 0.85}, '8': {'L': 12, 'A': 0.72}, '9': {'L': 13, 'A': 0.66}, '10': {'L': 14, 'A': 0.6}, '11': {'L': 15, 'A': 0.56}, '12': {'L': 16, 'A': 0.52}, '13': {'L': 17, 'A': 0.48}, '14': {'L': 18, 'A': 0.45}}, \
                 '1.2': {'4': {'L': 6, 'A': 1.37}, '5': {'L': 7, 'A': 1.27}, '6': {'L': 9, 'A': 1.05}, '7': {'L': 10, 'A': 0.93}, '8': {'L': 12, 'A': 0.79}, '9': {'L': 13, 'A': 0.72}, '10': {'L': 14, 'A': 0.66}, '11': {'L': 15, 'A': 0.61}, '12': {'L': 16, 'A': 0.56}, '13': {'L': 17, 'A': 0.53}, '14': {'L': 18, 'A': 0.49}}, \
                 '1.3': {'4': {'L': 6, 'A': 1.49}, '5': {'L': 7, 'A': 1.37}, '6': {'L': 9, 'A': 1.14}, '7': {'L': 10, 'A': 1.0}, '8': {'L': 12, 'A': 0.85}, '9': {'L': 13, 'A': 0.78}, '10': {'L': 14, 'A': 0.71}, '11': {'L': 15, 'A': 0.66}, '12': {'L': 16, 'A': 0.61}, '13': {'L': 17, 'A': 0.57}, '14': {'L': 18, 'A': 0.53}}, \
                 '1.4': {'4': {'L': 6, 'A': 1.6}, '5': {'L': 7, 'A': 1.48}, '6': {'L': 9, 'A': 1.22}, '7': {'L': 10, 'A': 1.08}, '8': {'L': 12, 'A': 0.92}, '9': {'L': 13, 'A': 0.84}, '10': {'L': 14, 'A': 0.77}, '11': {'L': 15, 'A': 0.71}, '12': {'L': 16, 'A': 0.66}, '13': {'L': 17, 'A': 0.61}, '14': {'L': 18, 'A': 0.58}}, \
                 '1.5': {'4': {'L': 6, 'A': 1.72}, '5': {'L': 7, 'A': 1.58}, '6': {'L': 9, 'A': 1.31}, '7': {'L': 10, 'A': 1.16}, '8': {'L': 12, 'A': 0.98}, '9': {'L': 13, 'A': 0.89}, '10': {'L': 14, 'A': 0.82}, '11': {'L': 15, 'A': 0.76}, '12': {'L': 16, 'A': 0.7}, '13': {'L': 17, 'A': 0.66}, '14': {'L': 18, 'A': 0.62}}, \
                 '1.6': {'4': {'L': 6, 'A': 1.83}, '5': {'L': 7, 'A': 1.69}, '6': {'L': 9, 'A': 1.4}, '7': {'L': 10, 'A': 1.23}, '8': {'L': 12, 'A': 1.05}, '9': {'L': 13, 'A': 0.95}, '10': {'L': 14, 'A': 0.88}, '11': {'L': 15, 'A': 0.81}, '12': {'L': 16, 'A': 0.75}, '13': {'L': 17, 'A': 0.7}, '14': {'L': 18, 'A': 0.66}}, \
                 '1.7': {'4': {'L': 6, 'A': 1.94}, '5': {'L': 7, 'A': 1.8}, '6': {'L': 9, 'A': 1.49}, '7': {'L': 10, 'A': 1.31}, '8': {'L': 12, 'A': 1.11}, '9': {'L': 13, 'A': 1.01}, '10': {'L': 14, 'A': 0.93}, '11': {'L': 15, 'A': 0.86}, '12': {'L': 16, 'A': 0.8}, '13': {'L': 17, 'A': 0.74}, '14': {'L': 18, 'A': 0.7}}, \
                 '1.8': {'4': {'L': 6, 'A': 1.99}, '5': {'L': 7, 'A': 1.9}, '6': {'L': 9, 'A': 1.57}, '7': {'L': 10, 'A': 1.39}, '8': {'L': 12, 'A': 1.18}, '9': {'L': 13, 'A': 1.07}, '10': {'L': 14, 'A': 0.98}, '11': {'L': 15, 'A': 0.91}, '12': {'L': 16, 'A': 0.84}, '13': {'L': 17, 'A': 0.79}, '14': {'L': 18, 'A': 0.74}}, \
                 '1.9': {'4': {'L': 6, 'A': 1.99}, '5': {'L': 7, 'A': 1.99}, '6': {'L': 9, 'A': 1.66}, '7': {'L': 10, 'A': 1.47}, '8': {'L': 12, 'A': 1.25}, '9': {'L': 13, 'A': 1.13}, '10': {'L': 14, 'A': 1.04}, '11': {'L': 15, 'A': 0.96}, '12': {'L': 16, 'A': 0.89}, '13': {'L': 17, 'A': 0.83}, '14': {'L': 18, 'A': 0.78}}}

    # the first half images are with binary signals (two dots)
    bn_sig = np.zeros([half_size, dim1, dim2])
    for sig_idx in range(half_size):
        loc_index = np.random.randint(32274)
        L_index = np.random.randint(len(doublet_L))
        while not np.any(np.all(loc == np.array([loc[loc_index, 0], loc[loc_index, 1] + doublet_L[L_index]]), axis=1)):
            loc_index = np.random.randint(32274)
            L_index = np.random.randint(len(doublet_L))
        bn_sig[sig_idx,:,:] = A * np.exp(-0.5 * ((X - loc[loc_index, 1]) ** 2 + (Y - loc[loc_index, 0]) ** 2) / (wid ** 2)) + \
                                A * np.exp(-0.5 * ((X - loc[loc_index, 1] - doublet_L[L_index]) ** 2 + (Y - loc[loc_index, 0]) ** 2) / (wid ** 2))
        # normalize the signal so that has the highest intensity as A
        bn_sig[sig_idx,:,:] = A * (bn_sig[sig_idx,:,:] - np.min(bn_sig[sig_idx,:,:])) / (np.max(bn_sig[sig_idx,:,:]) - np.min(bn_sig[sig_idx,:,:]))
        L_index_list[sig_idx] = L_index
        loc_list[sig_idx] = [loc[loc_index, 0], loc[loc_index, 1] - int(doublet_L[L_index] / 2)]

    # the second half images are with line signal
    line_sig = np.zeros([tot_size - half_size, dim1, dim2])
    for sig_idx in range(tot_size - half_size):
        loc_index = np.random.randint(32274)
        L_index = np.random.randint(len(doublet_L))
        cur_L = line_dict[str(A)][str(doublet_L[L_index])]['L']
        cur_A = line_dict[str(A)][str(doublet_L[L_index])]['A']

        while not np.any(np.all(loc == np.array([loc[loc_index, 0], loc[loc_index, 1] + cur_L]), axis=1)):
            loc_index = np.random.randint(32274)
            L_index = np.random.randint(len(doublet_L))
            cur_L = line_dict[str(A)][str(doublet_L[L_index])]['L']
            cur_A = line_dict[str(A)][str(doublet_L[L_index])]['A']
        # convolve over a line per Varun's SPIE paper
        for L_i in range(cur_L):
            line_sig[sig_idx,:,:] = line_sig[sig_idx,:,:] + \
                                    cur_A * np.exp(-0.5 * ((X - loc[loc_index, 1] - L_i) ** 2 + (Y - loc[loc_index, 0]) ** 2) / (wid ** 2))
        # normalize the signal so that has the highest intensity as A
        line_sig[sig_idx,:,:] = cur_A * (line_sig[sig_idx,:,:] - np.min(line_sig[sig_idx,:,:])) / (np.max(line_sig[sig_idx,:,:]) - np.min(line_sig[sig_idx,:,:]))
        L_index_list[half_size + sig_idx] = L_index
        loc_list[half_size +sig_idx] = [loc[loc_index, 0], loc[loc_index, 1] - int(cur_L / 2)]

    obj[:half_size,:,:] = obj[:half_size,:,:] + bn_sig
    obj[half_size:,:,:] = obj[half_size:,:,:] + line_sig

    return loc_list, L_index_list, obj


def AddSignalRayleigh_train(obj, wid, loc, point1, point2, point3, half_size, tot_size, dim1, dim2):
# Add Rayleigh signals (binary and line) to the input object for training purposes.
#
# Args:
#     obj (numpy.ndarray): Input object to add signals to.
#     wid (float): Width of the Gaussian signal.
#     loc (numpy.ndarray): Array of possible locations for signals.
#     point1, point2, point3 (float): Points defining the "L" shape empty area in the 2AFC HO table.
#     half_size (int): Half of the total number of objects.
#     tot_size (int): Total number of objects.
#     dim1, dim2 (int): Dimensions of each object.
#
# Returns:
#     numpy.ndarray: Object with added signals.

    As = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]  # np.arange(0.2, 2, 0.1)
    doublet_L = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # np.arange(4, 15, 1)
    np.random.seed(42)
    X, Y = np.meshgrid(range(dim2), range(dim1))

    half_size = int(half_size)
    tot_size = int(tot_size)
    L_index_list = np.zeros([tot_size, 1])
    loc_list = np.zeros([tot_size, 2])

    line_dict = {'0.1': {'4': {'L': 6, 'A': 0.11}, '5': {'L': 7, 'A': 0.11}, '6': {'L': 9, 'A': 0.09}, '7': {'L': 10, 'A': 0.08}, '8': {'L': 11, 'A': 0.07}, '9': {'L': 13, 'A': 0.06}, '10': {'L': 14, 'A': 0.05}, '11': {'L': 15, 'A': 0.05}, '12': {'L': 16, 'A': 0.05}, '13': {'L': 17, 'A': 0.04}, '14': {'L': 18, 'A': 0.04}}, \
                 '0.2': {'4': {'L': 6, 'A': 0.23}, '5': {'L': 7, 'A': 0.21}, '6': {'L': 9, 'A': 0.17}, '7': {'L': 10, 'A': 0.15}, '8': {'L': 12, 'A': 0.13}, '9': {'L': 13, 'A': 0.12}, '10': {'L': 14, 'A': 0.11}, '11': {'L': 15, 'A': 0.1}, '12': {'L': 16, 'A': 0.09}, '13': {'L': 17, 'A': 0.09}, '14': {'L': 18, 'A': 0.08}}, \
                 '0.3': {'4': {'L': 6, 'A': 0.34}, '5': {'L': 7, 'A': 0.32}, '6': {'L': 9, 'A': 0.26}, '7': {'L': 10, 'A': 0.23}, '8': {'L': 12, 'A': 0.2}, '9': {'L': 13, 'A': 0.18}, '10': {'L': 14, 'A': 0.16}, '11': {'L': 15, 'A': 0.15}, '12': {'L': 16, 'A': 0.14}, '13': {'L': 17, 'A': 0.13}, '14': {'L': 18, 'A': 0.12}}, \
                 '0.4': {'4': {'L': 6, 'A': 0.46}, '5': {'L': 7, 'A': 0.42}, '6': {'L': 9, 'A': 0.35}, '7': {'L': 10, 'A': 0.31}, '8': {'L': 12, 'A': 0.26}, '9': {'L': 13, 'A': 0.24}, '10': {'L': 14, 'A': 0.22}, '11': {'L': 15, 'A': 0.2}, '12': {'L': 16, 'A': 0.19}, '13': {'L': 17, 'A': 0.18}, '14': {'L': 18, 'A': 0.16}}, \
                 '0.5': {'4': {'L': 6, 'A': 0.57}, '5': {'L': 7, 'A': 0.53}, '6': {'L': 9, 'A': 0.44}, '7': {'L': 10, 'A': 0.39}, '8': {'L': 12, 'A': 0.33}, '9': {'L': 13, 'A': 0.3}, '10': {'L': 14, 'A': 0.27}, '11': {'L': 15, 'A': 0.25}, '12': {'L': 16, 'A': 0.23}, '13': {'L': 17, 'A': 0.22}, '14': {'L': 18, 'A': 0.21}}, \
                 '0.6': {'4': {'L': 6, 'A': 0.69}, '5': {'L': 7, 'A': 0.63}, '6': {'L': 9, 'A': 0.52}, '7': {'L': 10, 'A': 0.46}, '8': {'L': 12, 'A': 0.39}, '9': {'L': 13, 'A': 0.36}, '10': {'L': 14, 'A': 0.33}, '11': {'L': 15, 'A': 0.3}, '12': {'L': 16, 'A': 0.28}, '13': {'L': 17, 'A': 0.26}, '14': {'L': 18, 'A': 0.25}}, \
                 '0.7': {'4': {'L': 6, 'A': 0.8}, '5': {'L': 7, 'A': 0.74}, '6': {'L': 9, 'A': 0.61}, '7': {'L': 10, 'A': 0.54}, '8': {'L': 12, 'A': 0.46}, '9': {'L': 13, 'A': 0.42}, '10': {'L': 14, 'A': 0.38}, '11': {'L': 15, 'A': 0.35}, '12': {'L': 16, 'A': 0.33}, '13': {'L': 17, 'A': 0.31}, '14': {'L': 18, 'A': 0.29}}, \
                 '0.8': {'4': {'L': 6, 'A': 0.91}, '5': {'L': 7, 'A': 0.85}, '6': {'L': 9, 'A': 0.7}, '7': {'L': 10, 'A': 0.62}, '8': {'L': 12, 'A': 0.52}, '9': {'L': 13, 'A': 0.48}, '10': {'L': 14, 'A': 0.44}, '11': {'L': 15, 'A': 0.4}, '12': {'L': 16, 'A': 0.38}, '13': {'L': 17, 'A': 0.35}, '14': {'L': 18, 'A': 0.33}}, \
                 '0.9': {'4': {'L': 6, 'A': 0.99}, '5': {'L': 7, 'A': 0.95}, '6': {'L': 9, 'A': 0.79}, '7': {'L': 10, 'A': 0.69}, '8': {'L': 12, 'A': 0.59}, '9': {'L': 13, 'A': 0.54}, '10': {'L': 14, 'A': 0.49}, '11': {'L': 15, 'A': 0.45}, '12': {'L': 16, 'A': 0.42}, '13': {'L': 17, 'A': 0.39}, '14': {'L': 18, 'A': 0.37}}, \
                 '1.0': {'4': {'L': 6, 'A': 1.14}, '5': {'L': 7, 'A': 1.06}, '6': {'L': 9, 'A': 0.87}, '7': {'L': 10, 'A': 0.77}, '8': {'L': 12, 'A': 0.66}, '9': {'L': 13, 'A': 0.6}, '10': {'L': 14, 'A': 0.55}, '11': {'L': 15, 'A': 0.51}, '12': {'L': 16, 'A': 0.47}, '13': {'L': 17, 'A': 0.44}, '14': {'L': 18, 'A': 0.41}}, \
                 '1.1': {'4': {'L': 6, 'A': 1.26}, '5': {'L': 7, 'A': 1.16}, '6': {'L': 9, 'A': 0.96}, '7': {'L': 10, 'A': 0.85}, '8': {'L': 12, 'A': 0.72}, '9': {'L': 13, 'A': 0.66}, '10': {'L': 14, 'A': 0.6}, '11': {'L': 15, 'A': 0.56}, '12': {'L': 16, 'A': 0.52}, '13': {'L': 17, 'A': 0.48}, '14': {'L': 18, 'A': 0.45}}, \
                 '1.2': {'4': {'L': 6, 'A': 1.37}, '5': {'L': 7, 'A': 1.27}, '6': {'L': 9, 'A': 1.05}, '7': {'L': 10, 'A': 0.93}, '8': {'L': 12, 'A': 0.79}, '9': {'L': 13, 'A': 0.72}, '10': {'L': 14, 'A': 0.66}, '11': {'L': 15, 'A': 0.61}, '12': {'L': 16, 'A': 0.56}, '13': {'L': 17, 'A': 0.53}, '14': {'L': 18, 'A': 0.49}}, \
                 '1.3': {'4': {'L': 6, 'A': 1.49}, '5': {'L': 7, 'A': 1.37}, '6': {'L': 9, 'A': 1.14}, '7': {'L': 10, 'A': 1.0}, '8': {'L': 12, 'A': 0.85}, '9': {'L': 13, 'A': 0.78}, '10': {'L': 14, 'A': 0.71}, '11': {'L': 15, 'A': 0.66}, '12': {'L': 16, 'A': 0.61}, '13': {'L': 17, 'A': 0.57}, '14': {'L': 18, 'A': 0.53}}, \
                 '1.4': {'4': {'L': 6, 'A': 1.6}, '5': {'L': 7, 'A': 1.48}, '6': {'L': 9, 'A': 1.22}, '7': {'L': 10, 'A': 1.08}, '8': {'L': 12, 'A': 0.92}, '9': {'L': 13, 'A': 0.84}, '10': {'L': 14, 'A': 0.77}, '11': {'L': 15, 'A': 0.71}, '12': {'L': 16, 'A': 0.66}, '13': {'L': 17, 'A': 0.61}, '14': {'L': 18, 'A': 0.58}}, \
                 '1.5': {'4': {'L': 6, 'A': 1.72}, '5': {'L': 7, 'A': 1.58}, '6': {'L': 9, 'A': 1.31}, '7': {'L': 10, 'A': 1.16}, '8': {'L': 12, 'A': 0.98}, '9': {'L': 13, 'A': 0.89}, '10': {'L': 14, 'A': 0.82}, '11': {'L': 15, 'A': 0.76}, '12': {'L': 16, 'A': 0.7}, '13': {'L': 17, 'A': 0.66}, '14': {'L': 18, 'A': 0.62}}, \
                 '1.6': {'4': {'L': 6, 'A': 1.83}, '5': {'L': 7, 'A': 1.69}, '6': {'L': 9, 'A': 1.4}, '7': {'L': 10, 'A': 1.23}, '8': {'L': 12, 'A': 1.05}, '9': {'L': 13, 'A': 0.95}, '10': {'L': 14, 'A': 0.88}, '11': {'L': 15, 'A': 0.81}, '12': {'L': 16, 'A': 0.75}, '13': {'L': 17, 'A': 0.7}, '14': {'L': 18, 'A': 0.66}}, \
                 '1.7': {'4': {'L': 6, 'A': 1.94}, '5': {'L': 7, 'A': 1.8}, '6': {'L': 9, 'A': 1.49}, '7': {'L': 10, 'A': 1.31}, '8': {'L': 12, 'A': 1.11}, '9': {'L': 13, 'A': 1.01}, '10': {'L': 14, 'A': 0.93}, '11': {'L': 15, 'A': 0.86}, '12': {'L': 16, 'A': 0.8}, '13': {'L': 17, 'A': 0.74}, '14': {'L': 18, 'A': 0.7}}, \
                 '1.8': {'4': {'L': 6, 'A': 1.99}, '5': {'L': 7, 'A': 1.9}, '6': {'L': 9, 'A': 1.57}, '7': {'L': 10, 'A': 1.39}, '8': {'L': 12, 'A': 1.18}, '9': {'L': 13, 'A': 1.07}, '10': {'L': 14, 'A': 0.98}, '11': {'L': 15, 'A': 0.91}, '12': {'L': 16, 'A': 0.84}, '13': {'L': 17, 'A': 0.79}, '14': {'L': 18, 'A': 0.74}}, \
                 '1.9': {'4': {'L': 6, 'A': 1.99}, '5': {'L': 7, 'A': 1.99}, '6': {'L': 9, 'A': 1.66}, '7': {'L': 10, 'A': 1.47}, '8': {'L': 12, 'A': 1.25}, '9': {'L': 13, 'A': 1.13}, '10': {'L': 14, 'A': 1.04}, '11': {'L': 15, 'A': 0.96}, '12': {'L': 16, 'A': 0.89}, '13': {'L': 17, 'A': 0.83}, '14': {'L': 18, 'A': 0.78}}}

    # the first half images are with binary signals (two dots)
    bn_sig = np.zeros([half_size, dim1, dim2])
    for sig_idx in range(half_size):
        loc_index = np.random.randint(32274)
        L_index = np.random.randint(len(doublet_L))
        while not np.any(np.all(loc == np.array([loc[loc_index, 0], loc[loc_index, 1] + doublet_L[L_index]]), axis=1)):
            loc_index = np.random.randint(32274)
            L_index = np.random.randint(len(doublet_L))

        A_index = np.random.randint(len(As))
        A = As[A_index]

        # point1 (A), point2 (L), and point3 (L): three necessary points that can define the "L" shape empty area for each acc in the 2AFC HO table
        while A <= point1 and doublet_L[L_index] <= point2:
            if doublet_L[L_index] > point3:
                break
            A_index = np.random.randint(len(As))
            A = As[A_index]

        bn_sig[sig_idx,:,:] = A * np.exp(-0.5 * ((X - loc[loc_index, 1]) ** 2 + (Y - loc[loc_index, 0]) ** 2) / (wid ** 2)) + \
                                A * np.exp(-0.5 * ((X - loc[loc_index, 1] - doublet_L[L_index]) ** 2 + (Y - loc[loc_index, 0]) ** 2) / (wid ** 2))
        # normalize the signal so that has the highest intensity as A
        bn_sig[sig_idx,:,:] = A * (bn_sig[sig_idx,:,:] - np.min(bn_sig[sig_idx,:,:])) / (np.max(bn_sig[sig_idx,:,:]) - np.min(bn_sig[sig_idx,:,:]))
        L_index_list[sig_idx] = L_index
        loc_list[sig_idx] = [loc[loc_index, 0], loc[loc_index, 1] - int(doublet_L[L_index] / 2)]

    # the second half images are with line signal
    line_sig = np.zeros([tot_size - half_size, dim1, dim2])
    for sig_idx in range(tot_size - half_size):

        loc_index = np.random.randint(32274)
        L_index = np.random.randint(len(doublet_L))
        A_index = np.random.randint(len(As))
        A = As[A_index]

        cur_L = line_dict[str(A)][str(doublet_L[L_index])]['L']

        while not np.any(np.all(loc == np.array([loc[loc_index, 0], loc[loc_index, 1] + cur_L]), axis=1)):
            loc_index = np.random.randint(32274)
            L_index = np.random.randint(len(doublet_L))
            cur_L = line_dict[str(A)][str(doublet_L[L_index])]['L']

        # point1 (A), point2 (L), and point3 (L): three necessary points that can define the "L" shape empty area for each acc in the 2AFC HO table
        while A <= point1 and doublet_L[L_index] <= point2:
            if doublet_L[L_index] > point3:
                break
            A_index = np.random.randint(len(As))
            A = As[A_index]
            cur_L = line_dict[str(A)][str(doublet_L[L_index])]['L']

            while not np.any(np.all(loc == np.array([loc[loc_index, 0], loc[loc_index, 1] + cur_L]), axis=1)):
                loc_index = np.random.randint(32274)
                L_index = np.random.randint(len(doublet_L))
                cur_L = line_dict[str(A)][str(doublet_L[L_index])]['L']

        cur_A = line_dict[str(A)][str(doublet_L[L_index])]['A']

        # convolve over a line per Varun's SPIE paper
        for L_i in range(cur_L):
            line_sig[sig_idx,:,:] = line_sig[sig_idx,:,:] + \
                                    cur_A * np.exp(-0.5 * ((X - loc[loc_index, 1] - L_i) ** 2 + (Y - loc[loc_index, 0]) ** 2) / (wid ** 2))
        # normalize the signal so that has the highest intensity as A
        line_sig[sig_idx,:,:] = cur_A * (line_sig[sig_idx,:,:] - np.min(line_sig[sig_idx,:,:])) / (np.max(line_sig[sig_idx,:,:]) - np.min(line_sig[sig_idx,:,:]))
        L_index_list[half_size + sig_idx] = L_index
        loc_list[sig_idx] = [loc[loc_index, 0], loc[loc_index, 1] - int(cur_L / 2)]

    obj[:half_size,:,:] = obj[:half_size,:,:] + bn_sig
    obj[half_size:,:,:] = obj[half_size:,:,:] + line_sig

    return obj

def AddSignalDetection(obj, A, wid, loc, half_size, tot_size, dim1, dim2):
    np.random.seed(42)
    X, Y = np.meshgrid(range(dim2), range(dim1))

    half_size = int(half_size)
    tot_size = int(tot_size)

    sig = np.zeros([half_size, dim1, dim2])
    for sig_idx in range(half_size):
        loc_index = np.random.randint(32274)
        sig[sig_idx,:,:] = A * np.exp(-0.5 * ((X - loc[loc_index, 1]) ** 2 + (Y - loc[loc_index, 0]) ** 2) / (wid ** 2))

    obj[:half_size,:,:] = obj[:half_size,:,:] + sig

    return obj
