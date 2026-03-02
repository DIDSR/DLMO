
import os
import numpy as np

def list_all_npy_files(super_path, cmpr_dtype=None, unity_normalize=False):
    arr_accumulator = []
    idx = 0
    for main_folder in os.listdir(super_path): #outermost folder
        for sub_folder in os.listdir(os.path.join(super_path, main_folder)): # sub folders
            inner_path= os.path.join(super_path, main_folder, sub_folder) # each inner folder
            for file in os.listdir(inner_path):
                if file.endswith('.npz'):
                    each_npz_file = os.path.join(inner_path, file)
                    each_npz_read = np.load(each_npz_file)
                    each_npz_arr  = each_npz_read.f.arr_0
                    each_npz_arr  = np.squeeze(each_npz_arr)

                    # change the datatype and normalize the 'uint8'
                    if cmpr_dtype!=None: each_npz_arr = each_npz_arr.astype(cmpr_dtype)
                    if unity_normalize:  each_npz_arr = each_npz_arr/255.0
                    if idx ==0:
                        # print('------------------------')
                        # print('the npz file read are:')
                        # print('-------------------------')
                        arr_accumulator = each_npz_arr
                    else:
                        arr_accumulator = np.append (arr_accumulator, each_npz_arr, axis=0)
                    # print(each_npz_file)
                    idx = idx +1
    # print()
    # print('---------------------------------------------------------------------------')
    # print('size of array after reading all the npz files is:', arr_accumulator.shape)
    # print('---------------------------------------------------------------------------')
    # print()
    return(arr_accumulator)

