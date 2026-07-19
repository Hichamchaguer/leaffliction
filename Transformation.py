import sys
from plantcv import plantcv as pcv
import argparse
from pathlib import Path
import os


class Options:
    def __init__(self, image, dst=None, specific=None):
        self.image = image  # The path to the image.
        self.name = self.image.split("/")[-1]
        if dst:
            self.outdir = "./" + dst
            pcv.params.debug_outdir = "./" + dst
            self.debug = "print"  # Save the image or plot it.
        else:
            self.debug = "plot"
        if specific:
            self.specific = specific
        else:
            self.specific = "all"


class ImageTransformation:
    def __init__(self, image, dest, opt) -> None:
        self.pathname = image
        self.dest = dest
        self.image, self.path, self.filename = pcv.readimage(filename=image)
        self.opt = opt
        self._m_blur = None
        self._blur = None
        self._mask1 = None
        self._ab_fill = None
        self._mask = None
        self._roi_objects = None
        self._hierarchy3 = None
        self._kept_mask = None
        self._obj_area = None
        self._obj = None
        self._mask2 = None
        self._analyze = None
        self._colors = None
        self._landmarks = None
        pcv.params.debug = None

    def to_threshold(self):
        s = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")
        s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=85, max_value=255, object_type="light"
        )
        return s_thresh

    def original(self):
        if self.opt.specific == "all" or self.opt.specific == "original":
            pcv.params.debug = self.opt.debug
        image, path, filename = pcv.readimage(self.pathname)
        pcv.params.debug = None
        return image

    def m_blur(self):
        s_thresh = self.to_threshold()
        s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
        self._m_blur = s_mblur
        return s_mblur

    def blur(self):
        s_thresh = self.to_threshold()
        if self.opt.specific == "all" or self.opt.specific == "blur":
            pcv.params.debug = self.opt.debug
        gaussian_img = pcv.gaussian_blur(
            img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
        )
        pcv.params.debug = None
        self._blur = gaussian_img
        return gaussian_img

    def mask1(self):
        b = pcv.rgb2gray_lab(rgb_img=self.image, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b, threshold=160, max_value=255, object_type="light"
        )
        bs = pcv.logical_or(bin_img1=self._m_blur, bin_img2=b_thresh)
        masked = pcv.apply_mask(img=self.image, mask=bs, mask_color="white")
        self._mask1 = masked
        return masked

    def ab_fill(self):
        masked_a = pcv.rgb2gray_lab(rgb_img=self._mask1, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=self._mask1, channel="b")
        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, max_value=255, object_type="dark"
        )
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a,
            threshold=135,
            max_value=255,
            object_type="light")
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b,
            threshold=128,
            max_value=255,
            object_type="light")
        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
        ab_fill = pcv.fill(bin_img=ab, size=200)
        self._ab_fill = ab_fill
        return ab_fill

    def mask(self):
        if self.opt.specific == "all" or self.opt.specific == "mask":
            pcv.params.debug = self.opt.debug
        masked2 = pcv.apply_mask(
            img=self._mask1, mask=self._ab_fill, mask_color="white"
        )
        pcv.params.debug = None
        self._mask = masked2
        return masked2

    def roi(self):
        id_objects, obj_hierarchy = pcv.find_objects(
            img=self._mask,
            mask=self._ab_fill)
        roi1, roi_hierarchy = pcv.roi.rectangle(
            img=self._mask,
            x=0,
            y=0,
            h=256,
            w=256)
        if self.opt.specific == "all" or self.opt.specific == "roi":
            pcv.params.debug = self.opt.debug
        roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(
            img=self.image,
            roi_contour=roi1,
            roi_hierarchy=roi_hierarchy,
            object_contour=id_objects,
            obj_hierarchy=obj_hierarchy,
            roi_type="partial",
        )
        pcv.params.debug = None
        self._roi_objects = roi_objects
        self._hierarchy3 = hierarchy3
        self._kept_mask = kept_mask
        self._obj_area = obj_area
        return roi_objects, hierarchy3, kept_mask, obj_area

    def obj_composition(self):
        obj, mask = pcv.object_composition(
            img=self.image,
            contours=self._roi_objects,
            hierarchy=self._hierarchy3)
        self._obj = obj
        self._mask2 = mask
        return obj, mask

    def analyze(self):
        if self.opt.specific == "all" or self.opt.specific == "analyze":
            pcv.params.debug = self.opt.debug
        analysis_image = pcv.analyze_object(
            img=self.image, obj=self._obj, mask=self._mask2, label="default"
        )
        pcv.params.debug = None
        return analysis_image

    def colors(self):
        if self.opt.specific == "all" or self.opt.specific == "colors":
            pcv.params.debug = self.opt.debug
        color_histogram = pcv.analyze_color(
            rgb_img=self.image,
            mask=self._kept_mask,
            colorspaces="all",
            label="default")
        pcv.params.debug = None
        self._colors = color_histogram
        return color_histogram

    def landmarks(self):
        if self.opt.specific == "all" or self.opt.specific == "landmarks":
            pcv.params.debug = self.opt.debug
        top_x, bottom_x, center_v_x = pcv.x_axis_pseudolandmarks(
            img=self.image, obj=self._obj, mask=self._mask2, label="default"
        )
        pcv.params.debug = None

    def apply_transformation(self):
        self.original()
        self.m_blur()
        self.blur()
        self.mask1()
        self.ab_fill()
        self.mask()
        self.roi()
        self.obj_composition()
        self.analyze()
        self.landmarks()
        self.colors()


def get_specific_transformation(args):
    specific = None
    if args.original:
        specific = "original"
    elif args.blur:
        specific = "blur"
    elif args.mask:
        specific = "mask"
    elif args.roi:
        specific = "roi"
    elif args.analyze:
        specific = "analyze"
    elif args.landmarks:
        specific = "landmarks"
    elif args.colors:
        specific = "colors"
    return specific


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This program transform image with
        multiples transformation techniques.
        You can see the different ways and option
        to use the programe correctly."""
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-original",
        help="Get the original picture.",
        action="store_true")
    group.add_argument(
        "-blur",
        help="Transform the image with a gaussian blur.",
        action="store_true")
    group.add_argument(
        "-mask",
        help="Print the mask of the leaf.",
        action="store_true")
    group.add_argument(
        "-roi",
        help="Show the ROI (Region of interest) of the picture.",
        action="store_true",
    )
    group.add_argument(
        "-analyze",
        help="Show the analyzed image. ROI + width and height.",
        action="store_true",
    )
    group.add_argument(
        "-landmarks",
        help="Show the pseudolandmarks on the leaf.ves.",
        action="store_true",
    )
    group.add_argument(
        "-colors",
        help="Print the repartition diagram of colors on the image.",
        action="store_true",
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        help="The path of a single image.")
    parser.add_argument(
        "-src",
        nargs=1,
        help="Transform all images in a folder (with specific tranformation)",
        type=str,
    )
    parser.add_argument(
        "-dst",
        nargs=1,
        help="Put the transformed images in the destination folder.",
        type=str,
    )
    args = parser.parse_args(sys.argv[1::])

    specific = get_specific_transformation(args)

    if args.src and not args.dst:
        print("You must specify the destination folder when -src flag is up")
        exit(1)
    elif args.src and args.path:
        print(
            "You must have only the path of"
            + " an image or the path of a folder")
        exit(1)
    if args.dst:
        dst = Path(args.dst[0])
        if not dst.exists():
            os.makedirs("./" + args.dst[0])
        if args.dst[0][-1] == "/":
            dest_path = args.dst[0]
        else:
            dest_path = args.dst[0] + "/"
    if args.src:
        src = Path(args.src[0])
        if not src.exists() or not src.is_dir():
            print("This folder does not exist.")
            exit(1)
        for x in src.iterdir():
            if x.is_dir():
                print(
                    "The path is wrong,"
                    + "You must be in the last subfolder"
                    + "containing all the images."
                )
                exit(1)
        for x in src.iterdir():
            image_path = str(x).split("/")[-1] + "/"
            dest_subpath = dest_path + image_path
            check_dst_path = Path(dest_subpath)
            if not check_dst_path.exists():
                os.makedirs(dest_subpath)
            opt = Options(str(x), dst=dest_subpath, specific=specific)
            im = ImageTransformation(str(x), dest=args.dst, opt=opt)
            im.apply_transformation()
    else:
        path = Path(args.path)
        if not path.exists() or not path.is_file():
            print("This file does not exist.")
            exit(1)
        dst = None
        if args.dst:
            dst = str(args.dst[0])
        opt = Options(str(args.path), dst=dst, specific=specific)
        im = ImageTransformation(str(args.path), dest=dst, opt=opt)
        im.apply_transformation()