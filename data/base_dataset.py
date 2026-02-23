from abc import ABC, abstractmethod
import jittor as jt
from jittor.dataset import Dataset

# 继承自torch.utils.data.Dataset和ABC（抽象基类），是一个抽象类
class BaseDataset(Dataset, ABC):
    def __init__(self, opt, split, dataset_name):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.split = split
        self.root = opt.dataroot
        self.dataset_name = dataset_name.lower() # 数据集名称小写化
        self.set_attrs(total_len=0)

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass
