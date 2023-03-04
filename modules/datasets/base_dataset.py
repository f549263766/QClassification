import copy
import os.path as osp
import mmcv
import numpy as np
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import List
from .pipelines import Compose


def expanduser(path):
    """# Expand the user's home directory.
    Args:
        path (str): The path of annotation file.
    Return:
        :str: The path after expanduser.
    """
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


# ABCMeta: 多态类函数
# abs.abstractmethod 控制子类必须实现该方法
class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.
    Args:
        data_path_prefix (str, required): The prefix of data path.
        pipeline (list, required): A list of dict, where each element represents
            a operation defined in `datasets.pipelines`.
        classes (Sequence[str] | str | None): If classes is None, use
            default CLASSES defined by builtin dataset. If classes is a
            string, take it as a file name. The file contains the name of
            classes where each line contains one class name. If classes is
            a tuple or list, override the CLASSES defined by the dataset.
        ann_file (str | None, optional): The annotation file. Default to None.
            When ann_file is str,  the subclass is expected to read from the ann_file.
            When ann_file is None, the subclass is expected to read according to data_prefix
        test_mode (bool, optional): In train mode or test mode. Default to False.
    """

    CLASSES = None

    def __init__(self,
                 data_path_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        super(BaseDataset, self).__init__()
        # initialize intra-class variables
        self.data_path_prefix = data_path_prefix
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.ann_file = expanduser(ann_file)
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Return:
            :dict: mapping from class name to class index.
        """

        return {class_name: i for i, class_name in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Return:
            :np.ndarray: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])

        return gt_labels

    def get_category_ids(self, idx):
        """Get category id by index.
        Args:
            idx (int, required): Index of data.
        Return:
            :list[int]: Image category of specified index.
        """

        return [int(self.data_infos[idx]['gt_label'])]

    def prepare_data(self, idx):
        """Use transform for data pre-processing.
        Args:
            idx (int, required): Index of data.
        Return:
            :callable: The data with pipeline.
        """
        results = copy.deepcopy(self.data_infos[idx])

        return self.pipeline(results)

    def __len__(self):
        """Get the length of dataset.

        Return:
            :int: The length of dataset.
        """

        return len(self.data_infos)

    def __getitem__(self, idx):
        """Index data through subscripts.
        Args:
            idx (int, required): Index of data.
        Return:
            :data: Indexed data
        """

        return self.prepare_data(idx)

    # classmethod 无需实例化类即可调用该函数
    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use default CLASSES defined by builtin dataset.
                If classes is a string, take it as a file name. The file contains the name of classes where each line
                contains one class name. If classes is a tuple or list, override the CLASSES defined by the dataset.
        Returns:
            :tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # load a text file and parse the content as a list of strings.
            class_names = mmcv.list_from_file(expanduser(classes))
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names
