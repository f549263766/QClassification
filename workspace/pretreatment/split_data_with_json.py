import argparse
import os
import os.path as osp
import random
import re
import mmcv
import shutil
import tqdm
import time
from concurrent import futures
from modules.utlis import save_json, get_root_logger


def config_parse():
    """Input config from cmd line
    Return:
        :obj: 'parses.parse_args()': The dict of namespace for config.
    """
    parses = argparse.ArgumentParser("split data with json")
    parses.add_argument("--data_root", type=str, help="Dataset of path to load")
    parses.add_argument("--output_root", type=str, help="Path of output data")
    parses.add_argument("--split_ratio", type=float, default=0.8, help="Proportion of data set partition")
    parses.add_argument("--seed", type=int, default=2023, help="Setting of random seeds")
    parses.add_argument("--num_thread", type=int, default=20, help="Set the number of multi-threaded threads")
    configs = parses.parse_args()
    return configs


def copy_and_write(idx, category, data_list, img):
    """Copy image file and write info to json file
    Args:
        idx (int, required): Label index value.
        category (str, required): Label name of the dataset
        data_list (list, required): Data list.
        img (str, required): Path of original image file.
    """
    image_filename = osp.basename(img)
    output_root = osp.join(args.output_root, category, "image")
    mmcv.mkdir_or_exist(output_root)
    shutil.copyfile(img, osp.join(output_root, image_filename))
    data_list.append(dict(
        filename=image_filename,
        category=category,
        label=idx
    ))


def read_sub_dir(idx, sub_dir_path):
    """Read image data from category path.
    Args:
        idx (int, required): Label index value.
        sub_dir_path (str, required): Path of image data.
    """
    category = osp.basename(sub_dir_path)
    image_list = [osp.join(sub_dir_path, img_file) for img_file in os.listdir(sub_dir_path)
                  if re.search(r'.jpg|.png|.jpeg', img_file)]
    random.shuffle(image_list)
    split_length = int(len(image_list) * args.split_ratio)
    logger.info(f"{category}: train set: {split_length}, val set: {len(image_list) - split_length}")

    # multi-threaded processing training set and verification set
    with futures.ThreadPoolExecutor(max_workers=args.num_thread) as t:
        task_list = [t.submit(lambda p: copy_and_write(*p), [idx, category, train_list, train_img])
                     for train_img in image_list[:split_length]]
        for task in tqdm.tqdm(futures.as_completed(task_list), total=len(task_list)):
            task.result()

    with futures.ThreadPoolExecutor(max_workers=args.num_thread) as t:
        task_list = [t.submit(lambda p: copy_and_write(*p), [idx, category, val_list, val_img])
                     for val_img in image_list[split_length:]]
        for task in tqdm.tqdm(futures.as_completed(task_list), total=len(task_list)):
            task.result()


def read_data_root():
    """Read image data from root by each category.
    """
    sub_dir_list = [osp.join(args.data_root, sub_dir) for sub_dir in os.listdir(args.data_root)
                    if osp.isdir(osp.join(args.data_root, sub_dir))]

    for idx, sub_dir in enumerate(sub_dir_list):
        read_sub_dir(idx, sub_dir)


def main():
    read_data_root()


if __name__ == '__main__':
    tic = time.time()
    args = config_parse()
    random.seed(args.seed)
    timestamp = str(time.strftime('%Y%m%d', time.localtime()))
    logger = get_root_logger()
    train_list, val_list = [], []

    main()

    logger.info(f'total train set: {len(train_list)}, total val set: {len(val_list)}')
    train_filename = osp.join(args.output_root, f"train_set_{len(train_list)}_{timestamp}.json")
    val_filename = osp.join(args.output_root, f"val_set_{len(val_list)}_{timestamp}.json")
    save_json(train_filename, train_list)
    save_json(val_filename, val_list)

    elapsed = time.time() - tic
    print(f'time cost: {elapsed // 60} minutes.')
