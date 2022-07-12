from torch.utils.data import Dataset
import numpy as np

class szlDataset(Dataset):
    """
    深肿瘤的数据，从提前存好的npy文件中加载。
    每次将一个wsi的数据打包为[patch_num, 3 ,h, w]的格式
    """
    def __init__(self, path, transforms) -> None:
        super(szlDataset, self).__init__()
        self.transforms = transforms
        
    def __getitem__(self, index):
        save_dict_path = 'C:/Users/86136/Desktop/new_gcn/wsi_dict.npy'
        self.wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
        return self.wsi_dict[str(index)]
    def __len__(self):
        pass