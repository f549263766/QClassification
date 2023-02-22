import argparse
import os
import os.path as osp
import re
import time
import random
import mmcv
import tqdm
import shutil
from concurrent import futures
from modules.utlis import set_random_seed, get_root_logger


def config_parse():
    """Input config from cmd line.

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


def write_txt_file(file_path, data_list):
    """Write data list to txt file.
    Args:
        file_path (str, required): Path of output txt file.
        data_list (list, required): List of data written.
    """
    with open(file_path, "w", encoding='utf-8') as f:
        f.write("\n".join(data_list))


def copyfile(img_file, category, label, data_list, train=True):
    """Copy image file to output path.
    Args:
        img_file (str, required): Path of original image file.
        category (str, required): Label name of the dataset
        label (int, required): Label index value.
        data_list (list, required): List of data written.
        train (bool, optional): Is it a training set. Default to True.
    """
    if train:
        copy_root = osp.join(args.output_root, "train")
    else:
        copy_root = osp.join(args.output_root, "val")
    category_root = osp.join(copy_root, category)
    mmcv.mkdir_or_exist(copy_root)
    mmcv.mkdir_or_exist(category_root)

    output_path = osp.join(category_root, osp.basename(img_file))
    shutil.copyfile(img_file, output_path)

    data_list.append(" ".join([output_path, str(label)]))


def read_sub_dir(label, dir_path):
    """Read Read image data from category path.
    Args:
        label (int, required): Label index value.
        dir_path (str, required): Path of image data.
    """
    category = osp.basename(dir_path)
    image_list = [osp.join(dir_path, img) for img in os.listdir(dir_path)
                  if re.search(r".jpg|.png|.jpeg", img)]
    random.shuffle(image_list)
    split_length = int(len(image_list) * args.split_ratio)
    logger.info(f"{category}: train set: {split_length}, val set: {len(image_list) - split_length}")

    # multi-threaded processing training set and validation set
    with futures.ThreadPoolExecutor(max_workers=args.num_thread) as t:
        task_list = [t.submit(lambda p: copyfile(*p), [train_img_file, category, label, train_list, True])
                     for train_img_file in image_list[:split_length]]
        for task in tqdm.tqdm(futures.as_completed(task_list), total=len(task_list)):
            task.result()

    with futures.ThreadPoolExecutor(max_workers=args.num_thread) as t:
        task_list = [t.submit(lambda p: copyfile(*p), [val_img_file, category, label, val_list, False])
                     for val_img_file in image_list[split_length:]]
        for task in tqdm.tqdm(futures.as_completed(task_list), total=len(task_list)):
            task.result()

    class_list.append(category)


def read_image_data():
    """Read image data from root by each category.
    """
    sub_dir_list = [osp.join(args.data_root, sub_dir) for sub_dir in os.listdir(args.data_root)
                    if osp.isdir(osp.join(args.data_root, sub_dir))]

    for idx, sub_dir in enumerate(sub_dir_list):
        read_sub_dir(idx, sub_dir)


def main():
    read_image_data()


if __name__ == '__main__':
    tic = time.time()
    args = config_parse()
    set_random_seed(args.seed)
    logger = get_root_logger()

    timestamp = str(time.strftime('%Y%m%d', time.localtime()))
    train_list, val_list, class_list = [], [], []

    main()

    logger.info(f'the number of category: {len(class_list)}')
    logger.info(f'total train set: {len(train_list)}, total val set: {len(val_list)}')

    meta_root = osp.join(args.output_root, "meta")
    mmcv.mkdir_or_exist(meta_root)
    train_txt_file = osp.join(meta_root, "train.txt")
    val_txt_file = osp.join(meta_root, "val.txt")
    classes_txt_file = osp.join(meta_root, "classes.txt")
    write_txt_file(train_txt_file, train_list)
    write_txt_file(val_txt_file, val_list)
    write_txt_file(classes_txt_file, class_list)

    elapsed = time.time() - tic
    print(f'time cost: {elapsed // 60} minutes.')
