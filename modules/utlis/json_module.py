import json

from modules.utlis.logger import get_root_logger


def load_json(json_file, start=0, end=-1, interval=50000):
    """Load json file
    Args:
        json_file (str, required): Json file of path for loading.
        start (int, optional): Read file start in this row. Defaults to 0.
        end (int, optional): Read file end in this row. Defaults to -1.
        interval (int, optional): Interval between printing information. Defaults to 50000.
    Returns:
        :list: 'data_list': The obtained data list.
    """
    data_list = []
    with open(json_file, 'r') as fid:
        for i, line in enumerate(fid):
            try:
                if end != -1 and i >= end:
                    break
                if i >= start:
                    line = line.strip()
                    data = json.loads(line)
                    data_list.append(data)
                if (i + 1) % interval == 0:
                    logger.info(f'already loaded {i + 1} items')
            except Exception as e:
                print(f'line: {line}, {e}')
    return data_list


def save_json(json_file, data_list, interval=50000):
    """Save json file
        Args:
            json_file (str, required): Json file of path for saving.
            data_list (list, required): Data list for saving to json file.
            interval (int, optional): Interval between printing information. Defaults to 50000.
        """
    with open(json_file, 'w') as fid:
        for i, data in enumerate(data_list):
            try:
                fid.write(json.dumps(data, ensure_ascii=False) + '\n')
                if (i + 1) % interval == 0:
                    logger.info(f'already wrote {i + 1} items')
            except Exception as e:
                print(f'data: {data}, {e}')


logger = get_root_logger()
