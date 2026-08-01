import os
import sys


def check_extension(path: str) -> bool:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        bool: _description_
    """
    ext = (".jpg")
    if not path.lower().endswith(ext):
        return False
    return True


def check_img(src: str) -> bool:
    """_summary_

    Args:
        src (str): _description_
    """
    if not os.path.isfile(src) or not check_extension(src):
        print("This image doesn't exist or the file isn't a JGP file")
        return False
    return True


def check_dir(dr: str) -> None:
    """_summary_

    Args:
        path (str): _description_
    """
    c = False
    if os.path.isdir(dr):
        c = all([os.path.isfile(os.path.join(dr, i)) for i in os.listdir(dr)])
    if not os.path.isdir(dr) or not c:
        print("This directory doesn't existe Or")
        print("There is no images in your directory")
        sys.exit(0)
