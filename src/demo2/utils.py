
import os
import numpy as np
import matplotlib.pyplot as plt

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

def multi2dplots(nrows, ncols, fig_arr, axis, passed_fig_att=None):
    """
    customized function to show different layers of a 3D array 
    as 2D subplots
    
    usage
    ------
    multi2dplots(1, 2, lena_stack, axis=0, passed_fig_att={'colorbar': False, \
    'split_title': np.asanyarray(['a','b']),'out_path': 'last_lr.tif'})
    where lena_stack is of size (2, 512, 512)
    
    input
    -----
    nrows       : no. of rows in the subplots
    ncols       : no. of columns in the subplots
    fig_arr     : 3D array used for the subplots
    axis        : axis that is held constant and the 2D plot is demostrated
                  along the other two axes
    passed_fig_att : customized arguments imported from pyplot's subplot kwargs
                     See the default arguments below
                     
    output
    -----
    subplots as figure
    """
    default_att= {"suptitle": '',
            "split_title": np.asanyarray(['']*(nrows*ncols)),
            "supfontsize": 12,
            "xaxis_vis"  : False,
            "yaxis_vis"  : False,
            "out_path"   : '',
            "figsize"    : [8, 8],
            "cmap"       : 'Greys_r',
            "plt_tight"  : True,
            "colorbar"   : True
                 }
    if passed_fig_att is None:
        fig_att = default_att
    else:
        fig_att = default_att
        for key, val in passed_fig_att.items():
            fig_att[key]=val
    
    f, axarr = plt.subplots(nrows, ncols, figsize = fig_att["figsize"])
    img_ind  = 0
    f.suptitle(fig_att["suptitle"], fontsize = fig_att["supfontsize"])
    for i in range(nrows):
        for j in range(ncols):                
            if (axis==0):
                each_img = fig_arr[img_ind, :, :]
            if (axis==1):
                each_img = fig_arr[:, img_ind, :]
            if (axis==2):
                each_img = fig_arr[:, :, img_ind]
                
            if(nrows==1):
                ax = axarr[j]
            elif(ncols ==1):
                ax =axarr[i]
            else:
                ax = axarr[i,j]
            im = ax.imshow(each_img, cmap = fig_att["cmap"])
            if fig_att["colorbar"] is True:  f.colorbar(im, ax=ax)
            ax.set_title(fig_att["split_title"][img_ind])
            ax.get_xaxis().set_visible(fig_att["xaxis_vis"])
            ax.get_yaxis().set_visible(fig_att["yaxis_vis"])
            img_ind = img_ind + 1
            if fig_att["plt_tight"] is True: plt.tight_layout()
            
    if (len(fig_att["out_path"])==0):
        plt.show()
    else:
        plt.savefig(fig_att["out_path"])