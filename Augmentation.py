import os
import sys
import pandas as pd
from srcs.blance_data import flip, rotate, skew, shear, crop, distortion


def blance_data(__type: int, img_path: str, src: str) -> None:
    """_summary_

    Args:
        __type (int): _description_
        img_path (str): _description_
        src (str): _description_
    """
    blance = {
        0: flip,
        1: rotate,
        2: skew,
        3: shear,
        4: crop,
        5: distortion
    }
    if __type in blance:
        blance[__type](img_path, src)


def main():
    try:
        assert len(sys.argv) == 2, "Need only the directory path"
        __dir1 = sys.argv[1]
        __dir2 = 'analyze'

        for elem in os.listdir(__dir2):
            if not elem.endswith('.csv'):
                continue
            path = os.path.join(__dir2, elem)
            df = pd.read_csv(path)
            max_val = df.max().max()
            ind_max = df.max().idxmax()
            for elm in os.listdir(__dir1):
                path = os.path.join(__dir1, elm)
                __ck = elm.startswith(elem.split('.')[0])
                if not __ck or not os.path.isdir(path) or elm == ind_max:
                    continue
                val = df[elm][0]
                if val >= max_val:
                    break
                indx = 0
                images = sorted(os.listdir(path))
                for img in images:
                    if val >= max_val or len(images) >= max_val:
                        break
                    indx %= 6
                    while indx < 6 and val < max_val:
                        blance_data(indx, path, img)
                        indx += 1
                        val += 1

    except AssertionError as err:
        print(f"AssertionError: {err}")
    except Exception as err:
        print(f"Error exception {err} \nline: {err.__traceback__.tb_lineno}")


if __name__ == "__main__":
    main()
