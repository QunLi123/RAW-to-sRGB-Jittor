import importlib
#import torch.utils.data
import jittor as jt
from data.base_dataset import BaseDataset


# 接受一个str输入，
def find_dataset_using_name(dataset_name, split='train'):
    dataset_filename = "data." + dataset_name + "_dataset" # 拼凑文件名
    # "data.srraw_dataset" -> 对应文件路径 data/srraw_dataset.py
    datasetlib = importlib.import_module(dataset_filename) # 动态决定导入哪个，本质是凑了个str

    dataset = None
    # "srraw" -> "SrrawDataset" (忽略大小写)
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    # 遍历datasetlib的所有成员尝试找类
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of "
                        "BaseDataset with class name that matches %s in "
                        "lowercase." % (dataset_filename, target_dataset_name))
    return dataset


# 这个是对外接口，接受数据集名称、split和opt，返回一个dataset实例
def create_dataset(dataset_name, split, opt):
    data_loader = CustomDatasetDataLoader(dataset_name, split, opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, dataset_name, split, opt):
        self.opt = opt
        dataset_class = find_dataset_using_name(dataset_name, split) # 这一步只是找到类
        self.dataset = dataset_class(opt, split, dataset_name) # 这里才是实例化
        self.imio = self.dataset.imio
        print("dataset [%s(%s)] created" % (dataset_name, split))
        self.dataloader = jt.dataset.DataLoader(
            self.dataset,
            batch_size=opt.batch_size if split=='train' else 1,
            shuffle=opt.shuffle and split=='train',
            num_workers=int(opt.num_dataloader), 
            drop_last=opt.drop_last)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

