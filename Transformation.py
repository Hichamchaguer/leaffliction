import os
import sys
import argparse
from tqdm import tqdm
from srcs.tools import check_dir, check_img
from srcs.transform_image import ImageTransformation


def check_transformations(args: argparse.Namespace) -> list:
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        list[str | None]: _description_
    """
    lst = []

    if args.original:
        lst.append('original')
    if args.mask:
        lst.append('mask')
    if args.blur:
        lst.append('blur')
    if args.roi:
        lst.append('roi')
    if args.analyze:
        lst.append('analyze')
    if args.pseudo:
        lst.append('pseudo')
    if not len(lst):
        lst = ["original", "mask", "blur", "roi", "analyze", "pseudo"]
    return lst


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Apply image transformations (Gaussian blur, Mask, \
                         Roi objects, Analyze object, Pseudolandmarks)",
            epilog="Examples:\n  ./Transformation.py image.jpg\n  \
                    ./Transformation.py -src images/ -dst output/ -mask"
        )

        # group = parser.add_mutually_exclusive_group(required=True)
        parser.add_argument('image_path',
                            nargs='?',
                            help='Path to a single image file'
                            )
        parser.add_argument('-src',
                            help='Source directory containing images'
                            )
        parser.add_argument('-dst',
                            help='Destination directory for output'
                            )
        parser.add_argument('-blur', action='store_true',
                            help='Aplly Gaussian blur transformation'
                            )
        parser.add_argument('-mask', action='store_true',
                            help='Apply Mask transformation'
                            )
        parser.add_argument('-roi', action='store_true',
                            help='Apply ROI objects transformation'
                            )
        parser.add_argument('-analyze', action='store_true',
                            help='Apply Analyze object transformation'
                            )
        parser.add_argument('-pseudo', action='store_true',
                            help='Apply Pseudolandmarks transformation'
                            )
        parser.add_argument('-original', action='store_true',
                            help='Original images'
                            )

        args = parser.parse_args()

        if bool(args.image_path) == bool(args.src):
            parser.error("provide exactly one of: image_path OR -src")

        if args.image_path:
            if args.dst or args.mask:
                parser.error("-dst and -mask can only be used with -src, \
                             not with a single image path")
            if not check_img(args.image_path):
                sys.exit(1)
            single = ImageTransformation(args.image_path)
            single.transforme_single_image()
        elif args.src:
            if not args.dst:
                parser.error("-dst is missing")
            check_dir(args.src)
            if not os.path.isdir(args.dst):
                os.makedirs(args.dst, exist_ok=True)
            transformations = check_transformations(args)
            images = os.listdir(args.src)
            for img in tqdm(images, desc="Processing images", unit="image"):
                src = os.path.join(args.src, img)
                if not check_img(src):
                    continue
                trans = ImageTransformation(src)
                trans.dst = args.dst
                trans.transformations = transformations
                trans.apply_transformations()
                trans.save_transformations()

    except AssertionError as err:
        print(f"AssertionError: {err}")
    except Exception as err:
        print(f"Error exception {err} \nline: {err.__traceback__.tb_lineno}")


if __name__ == "__main__":
    main()
