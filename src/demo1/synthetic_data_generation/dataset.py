from torch.utils.data import Dataset
import numpy as np

class DatasetFromNpz(Dataset):
    def __init__(self, file_path):
        super(DatasetFromNpz, self).__init__()
        self.data  = np.load(file_path)
        #self.data = hf["arr_0"]

    def __getitem__(self, index):
        input = self.data[index,:,:,:]
        return input

    def __len__(self):
        return self.data.shape[0]