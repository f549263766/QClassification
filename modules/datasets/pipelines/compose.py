from collections.abc import Sequence
from mmcv.utils import build_from_cfg
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.
    Args:
        transforms (list[dict | callable], required): Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        # init transforms to list
        self.transforms = []
        # parse transforms
        for transform in transforms:
            if isinstance(transform, dict):
                # build a module from config dict when it is a class configuration,
                # or call a function from config dict when it is a function configuration.
                transform = build_from_cfg(transform, PIPELINES)  # return a instantiated object
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, but got {type(transform)}')

    def __call__(self, data):
        """Parse transforms with input data.
        Args:
            data (list[dict | callable], required): Either config dicts of transforms or transform objects.
        Return:
            data: Data after transforms operations.
        """
        for trans in self.transforms:
            data = trans(data)
            if data is None:
                return None

        return data

    def __repr__(self):
        """Print information of this class.

        Return:
            :str: The information of this class with format.
        """
        format_string = self.__class__.__name__ + '('
        for trans in self.transforms:
            format_string += f'\n    {trans}'
        format_string += '\n)'

        return format_string
