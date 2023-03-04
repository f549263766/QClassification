import gzip
import hashlib
import os
import os.path as osp
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile

__all__ = ['rm_suffix', 'check_integrity', 'download_and_extract_archive']


def rm_suffix(s, suffix=None):
    """Remove string suffix.
    Args:
        s (str, required): Enter the string to be processed.
        suffix (str, optional): Postfix Expression. Default to None.
    Return:
        :str: string with the specified suffix removed.
    """
    if suffix is None:
        return s[:s.rfind('.')]
    else:
        return s[:s.rfind(suffix)]


def calculate_md5(file_path, chunk_size=1024 * 1024):
    """Calculate the md5 value of the file.
    Args:
        file_path (str, required): Enter the path of the file to be calculated.
        chunk_size (int, optional): Read the size of binary file block. Default to 1024 * 1024.
    Return:
        :str: hash encrypted string.
    """
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)

    return md5.hexdigest()


def check_md5(file_path, md5, **kwargs):
    """Check whether the md5 value of the file is correct.
    Args:
        file_path (str, required): Enter the path of the file to be calculated.
        md5 (int, required): Md5 validation value.
    Return:
        :bool: is the file md5 value correct.
    """
    return md5 == calculate_md5(file_path, **kwargs)


def check_integrity(file_path, md5=None):
    """Check whether the file is complete by verifying the md5 value.
    Args:
        file_path (str, required): Enter the path of the file to be calculated.
        md5 (int, optional): Md5 validation value. Default to None.
    Return:
        :bool: is the file complete.
    """
    if not osp.isfile(file_path):
        return False
    if md5 is None:
        return True

    return check_md5(file_path, md5)


def download_url_to_file(url, file_path):
    """Download the file to the path through the url.
    Args:
        url (str, required): URL of the file.
        file_path (str, required): Path to save the downloaded file
    """
    with urllib.request.urlopen(url) as resp, open(file_path, 'wb') as of:
        shutil.copyfileobj(resp, of)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str, required): URL to download file from.
        root (str, required): Directory to place downloaded file in.
        filename (str | None, optional): Name to save the file under.
            If filename is None, use the basename of the URL.
        md5 (str | None, optional): MD5 checksum of the download.
            If md5 is None, download without md5 check.
    """
    root = osp.expanduser(root)
    if not filename:
        filename = osp.basename(url)
    file_path = osp.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(file_path, md5):
        print(f'Using downloaded and verified file: {file_path}')
    else:
        try:
            print(f'Downloading {url} to {file_path}')
            download_url_to_file(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      f' Downloading {url} to {file_path}')
                download_url_to_file(url, file_path)
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(file_path, md5):
            raise RuntimeError('File not found or corrupted.')


def extract_archive(from_path, to_path=None, remove_finished=False):
    """Extract compressed package file.
    Args:
        from_path (str, required): Path of compressed package to be decompressed.
        to_path (str, optional): Output path of compressed package decompression. Default to None.
        remove_finished (bool, optional): Whether to delete the compressed package after decompression.
            Default to False.
    """
    if to_path is None:
        to_path = osp.dirname(from_path)

    if from_path.endswith('.tar'):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith('.tar.gz') or from_path.endswith('.tgz'):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith('.tar.xz'):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith('.gz') and not from_path.endswith('.tar.gz'):
        to_path = os.path.join(
            to_path,
            os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, 'wb') as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif from_path.endswith('.zip'):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError(f'Extraction of {from_path} not supported')

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url,
                                 download_root,
                                 extract_root=None,
                                 filename=None,
                                 md5=None,
                                 remove_finished=False):
    """Download a file from a url and then extract compressed package file, place it in root.
    Args:
        url (str, required): URL to download file from.
        download_root (str, required): Directory to place downloaded file in.
        extract_root (str, required): Path of compressed package to be decompressed.
        filename (str | None, optional): Name to save the file under.
                If filename is None, use the basename of the URL.
        md5 (str | None, optional): MD5 checksum of the download.
                If md5 is None, download without md5 check.
        remove_finished (bool, optional): Whether to delete the compressed package after decompression.
                Default to False.
    """
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f'Extracting {archive} to {extract_root}')
    extract_archive(archive, extract_root, remove_finished)
