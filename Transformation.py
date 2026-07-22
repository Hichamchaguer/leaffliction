import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ioff()

from tqdm import tqdm

import cv2
import numpy as np
import argparse
from pathlib import Path
from plantcv import plantcv as pcv

class Options:
    def __init__(self, image, dst=None, specific=None):
        self.image = image
        self.name = Path(image).name

        if dst:
            self.outdir = dst
            self.debug = "print"
        else:
            self.debug = "plot"

        self.specific = specific if specific else "all"


class ImageTransformation:
    def __init__(self, image_path, dest=None, opt=None):
        self.pathname = str(image_path)
        self.dest = dest
        self.opt = opt or Options(image_path)

        self.image, self.path, self.filename = pcv.readimage(
            filename=self.pathname
        )

        self._s_thresh = None
        self._m_blur = None
        self._gaussian = None
        self._mask1 = None
        self._ab_fill = None
        self._final_mask_img = None
        self._labeled_mask = None
        self._color_fig = None

        self.vis = {}   

        pcv.params.debug = None

    def _to_rgb(self, img):
        """Shared helper — convert BGR/grayscale to RGB for display."""
        if img is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def original(self):
        self.vis["Original"] = self._to_rgb(self.image)
        return self.image
    def threshold(self):
        """
        HSV saturation channel -> binary threshold.
        Vivid green/yellow leaf tissue has HIGH saturation.
        Grey/white background has NEAR-ZERO saturation.
        """
        s = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")
        self._s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=85, object_type="light"
        )
        return self._s_thresh
    def m_blur(self):
        """Median blur removes salt-and-pepper noise from the threshold."""
        self._m_blur = pcv.median_blur(gray_img=self._s_thresh, ksize=5)
        return self._m_blur

    def blur(self):
        """Gaussian blur — the required display output (Figure IV.2)."""
        self._gaussian = pcv.gaussian_blur(
            img=self._s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
        )
        self.vis["Gaussian Blur"] = self._to_rgb(self._gaussian)
        return self._gaussian

    def mask1(self):
        """
        Combine saturation mask with LAB 'b' (blue-yellow) mask.
        Union catches both vivid-green leaf tissue and yellow disease
        spots that might have lower saturation.
        """
        b = pcv.rgb2gray_lab(rgb_img=self.image, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b, threshold=160, object_type="light"
        )
        combined = pcv.logical_or(bin_img1=self._m_blur, bin_img2=b_thresh)
        self._mask1 = pcv.apply_mask(
            img=self.image, mask=combined, mask_color="white"
        )
        return self._mask1
    def ab_fill(self):
        """
        Refine using LAB 'a' (green-magenta) and 'b' (blue-yellow)
        channels on the already-masked image, then remove blobs
        smaller than 200 pixels.
        """
        masked_a = pcv.rgb2gray_lab(rgb_img=self._mask1, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=self._mask1, channel="b")

        ta = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, object_type="dark"
        )
        ta1 = pcv.threshold.binary(
            gray_img=masked_a, threshold=135, object_type="light"
        )
        tb = pcv.threshold.binary(
            gray_img=masked_b, threshold=128, object_type="light"
        )

        combined = pcv.logical_or(bin_img1=ta, bin_img2=tb)
        combined = pcv.logical_or(bin_img1=ta1, bin_img2=combined)
        self._ab_fill = pcv.fill(bin_img=combined, size=200)
        return self._ab_fill

    def mask(self):
        """Apply the refined ab_fill mask onto mask1 — final clean image."""
        self._final_mask_img = pcv.apply_mask(
            img=self._mask1, mask=self._ab_fill, mask_color="white"
        )
        self.vis["Mask"] = self._to_rgb(self._final_mask_img)
        return self._final_mask_img
    def roi(self):
        """
        Define ROI covering the whole image, filter the mask down to
        a single labeled leaf object. Current PlantCV API: one call
        to pcv.roi.filter() replaces the older 3-function chain.
        """
        h, w = self._ab_fill.shape[:2]
        roi = pcv.roi.rectangle(img=self.image, x=0, y=0, h=h, w=w)
        self._labeled_mask = pcv.roi.filter(
            mask=self._ab_fill, roi=roi, roi_type="partial"
        )

        # build a visual: draw the kept contour outline in green
        vis = self.image.copy()
        contours, _ = cv2.findContours(
            (self._labeled_mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        self.vis["ROI Objects"] = self._to_rgb(vis)
        return self._labeled_mask
    def analyze(self):
        """
        Measure leaf shape: area, perimeter, convex hull, solidity.
        Produces an annotated outline image (Figure IV.5).
        """
        try:
            shape_img = pcv.analyze.size(
                img=self.image, labeled_mask=self._labeled_mask, n_labels=1
            )
            self.vis["Analyze Object"] = self._to_rgb(shape_img)
        except AttributeError:
            # older PlantCV versions use pcv.analyze_object instead
            print("  [analyze] pcv.analyze.size not found — "
                  "check your PlantCV version, may need pcv.analyze_object")
            self.vis["Analyze Object"] = self._to_rgb(self.image)
    def landmarks(self):
        """
        Place pseudolandmarks along top / bottom / center of leaf axis.
        Draw them in 3 distinct colors for the summary display.
        """
        vis = self.image.copy()
        try:
            top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
                img=self.image, mask=self._labeled_mask
            )
            groups = [(top, (0, 0, 255)),      # red   (BGR)
                      (bottom, (255, 0, 0)),    # blue  (BGR)
                      (center_v, (0, 255, 0))]  # green (BGR)
            for pts, colour in groups:
                for pt in pts:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(vis, (x, y), 4, colour, -1)
        except AttributeError:
            print("  [landmarks] pcv.homology.x_axis_pseudolandmarks not "
                  "found — check your PlantCV version")

        self.vis["Pseudolandmarks"] = self._to_rgb(vis)

    def colors(self):
        """
        Compute pixel-intensity histograms across 9 colour channels.
        We call pcv.analyze.color() only to populate
        pcv.outputs.observations, then plot the raw numbers ourselves —
        this avoids depending on whatever chart object PlantCV returns
        (matplotlib Figure in some versions, Altair chart in others).
        """
        try:
            pcv.analyze.color(
                rgb_img=self.image,
                labeled_mask=self._labeled_mask,
                n_labels=1,
                colorspaces="all",
            )
        except Exception as e:
            print(f"  [colors] analyze.color failed: {e}")
            self._color_fig = None
            return None

        self._color_fig = self._build_color_histogram_figure()
        return self._color_fig
 
    def _build_color_histogram_figure(self):
            obs = pcv.outputs.observations
            sample_key = None
            for key in obs.keys():
                if any(ch in obs[key] for ch in
                       ("hue_frequencies", "blue_frequencies")):
                    sample_key = key
                    break

            if sample_key is None:
                print("  [colors] could not find histogram data — "
                      "print(pcv.outputs.observations) to inspect keys")
                return None

            data = obs[sample_key]
            channels = [
                ("blue_frequencies",          "blue",    "blue"),
                ("green_frequencies",         "green",   "green"),
                ("red_frequencies",           "red",     "red"),
                ("lightness_frequencies",     "gray",    "lightness"),
                ("green-magenta_frequencies", "magenta", "green-magenta"),
                ("blue-yellow_frequencies",   "gold",    "blue-yellow"),
                ("hue_frequencies",           "purple",  "hue"),
                ("saturation_frequencies",    "cyan",    "saturation"),
                ("value_frequencies",         "orange",  "value"),
            ]

            fig, ax = plt.subplots(figsize=(9, 5))
            for obs_key, colour, label in channels:
                if obs_key not in data:
                    continue
                values = data[obs_key]["value"]
                if not values:
                    continue
                ax.plot(np.arange(len(values)), values,
                        color=colour, label=label, linewidth=1)

            ax.set_xlabel("Pixel intensity")
            ax.set_ylabel("Proportion of pixels (%)")
            ax.set_title("Colour Histogram", fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8, title="Colour Channel")
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            return fig
    def apply_transformation(self):
        self.original()
        self.threshold()
        self.m_blur()
        self.blur()
        self.mask1()
        self.ab_fill()
        self.mask()
        self.roi()
        self.analyze()
        self.landmarks()
        self.colors()

    def show_summary(self):
        order = ["Original", "Gaussian Blur", "Mask",
                 "ROI Objects", "Analyze Object", "Pseudolandmarks"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Transformation.py — {Path(self.pathname).name}",
                     fontsize=14, fontweight="bold")

        for ax, title in zip(axes.flatten(), order):
            img = self.vis.get(title)
            if img is None:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            ax.imshow(img)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        plt.show(block=True)

    def save_batch(self, dst_dir):
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        stem = Path(self.pathname).stem

        for title, img in self.vis.items():
            suffix = title.replace(" ", "_")
            out_path = dst / f"{stem}_{suffix}.jpg"
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr)

        if self._color_fig is not None:
            hist_path = dst / f"{stem}_Colors.jpg"
            self._color_fig.savefig(str(hist_path), bbox_inches="tight")
            plt.close(self._color_fig)
def get_specific_transformation(args):
    for flag in ("original", "blur", "mask", "roi",
                 "analyze", "landmarks", "colors"):
        if getattr(args, flag):
            return flag
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply a PlantCV transformation pipeline to a leaf image."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-original", action="store_true")
    group.add_argument("-blur", action="store_true")
    group.add_argument("-mask", action="store_true")
    group.add_argument("-roi", action="store_true")
    group.add_argument("-analyze", action="store_true")
    group.add_argument("-landmarks", action="store_true")
    group.add_argument("-colors", action="store_true")

    parser.add_argument("path", type=str, nargs="?",
                         help="Path to a single image file.")
    parser.add_argument("-src", nargs=1, type=str,
                         help="Source folder for batch processing.")
    parser.add_argument("-dst", nargs=1, type=str,
                         help="Destination folder (required with -src).")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    specific = get_specific_transformation(args)

    if args.src and not args.dst:
        print("ERROR: -dst is required when -src is used.")
        sys.exit(1)
    if args.src and args.path:
        print("ERROR: provide either a single image path OR -src, not both.")
        sys.exit(1)
    if not args.src and not args.path:
        parser.print_help()
        sys.exit(1)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    if args.src:
        src = Path(args.src[0])
        dst = Path(args.dst[0])
        if not src.exists() or not src.is_dir():
            print(f"ERROR: '{src}' is not a valid directory.")
            sys.exit(1)

        images = [p for p in src.iterdir()
                  if p.is_file() and p.suffix in IMAGE_EXTS]
        print(f"Processing {len(images)} images -> {dst}")

        for img_path in tqdm(images, desc="Transforming", unit="img"):
            opt = Options(str(img_path), dst=str(dst), specific=specific)
            transformer = ImageTransformation(img_path, dest=dst, opt=opt)
            transformer.apply_transformation()
            transformer.save_batch(dst)

    else:
        path = Path(args.path)
        if not path.exists() or not path.is_file():
            print(f"ERROR: '{path}' does not exist or is not a file.")
            sys.exit(1)

        opt = Options(str(path), dst=None, specific=specific)
        transformer = ImageTransformation(path, dest=None, opt=opt)
        transformer.apply_transformation()
        transformer.show_summary()


if __name__ == "__main__":
    main()





# test the blur 
# it = ImageTransformation("leaves/images/Apple_Black_rot/image (569).JPG")
# it.original()
# result = it.threshold()

# plt.imshow(result, cmap="gray")
# plt.title("Saturation Threshold")
# plt.show()
# it.threshold()
# mblur = it.m_blur()
# gblur = it.blur()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(mblur, cmap="gray")
# ax1.set_title("Median Blur")
# ax2.imshow(gblur, cmap="gray")
# ax2.set_title("Gaussian Blur")
# plt.show()


#threshold

# it = ImageTransformation("leaves/images/Apple_Black_rot/image (569).JPG")
# it.original()
# result = it.threshold()

# plt.imshow(result, cmap="gray")
# plt.title("Saturation Threshold")
# plt.show()

#test mask
# print("hols")

# it = ImageTransformation("leaves/images/Apple_Black_rot/image (569).JPG")
# it.original()
# it.threshold()   # compute _s_thresh
# it.m_blur()      # compute _m_blur
# it.mask1()       # now _m_blur exists and has the right shape
# plt.imshow(it._to_rgb(it._mask1))
# plt.title("Mask 1 (HSV + LAB-b combined)")
# print("hols")
# plt.show()

# it.ab_fill()
# plt.imshow(it._ab_fill, cmap="gray")
# plt.title("Refined mask (ab_fill)")
# plt.show()

# it.mask()

# plt.imshow(it.vis["Mask"])
# plt.title("Final Mask")
# plt.show()


#test roi

# it = ImageTransformation("leaves/images/Apple_Black_rot/image (569).JPG")
# it.original()
# it.threshold()   # compute _s_thresh
# it.m_blur()      # compute _m_blur
# it.mask1()       # now _m_blur exists and has the right shape
# it.ab_fill()
# it.roi()

# plt.imshow(it.vis["ROI Objects"])
# plt.title("ROI Objects")
# plt.show()

# print("Labeled mask shape:", it._labeled_mask.shape)
# print("Non-zero pixels:", (it._labeled_mask > 0).sum())
# it.analyze()

# plt.imshow(it.vis["Analyze Object"])
# plt.title("Analyze Object")
# plt.show()

# # print(pcv.outputs.observations)  
# it.landmarks()

# plt.imshow(it.vis["Pseudolandmarks"])
# plt.title("Pseudolandmarks")
# plt.show()

# fig = it.colors()
# fig.show()
# plt.show()

# it.apply_transformation()
# print(list(it.vis.keys()))