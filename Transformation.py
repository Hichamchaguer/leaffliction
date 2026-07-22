import argparse
from pathlib import Path
import sys
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt
import numpy as np


def analyze_saturation_correctly(img):
    """
    Correctly analyze saturation histogram to find the true valley
    """
    # Extract saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    
    # Create histogram
    hist, bins = np.histogram(s.flatten(), bins=256, range=(0, 256))
    
    # Find the valley (minimum) in the low range (0-100)
    # Looking for the dip that separates background from leaf
    low_range = hist[0:100]  # Focus on 0-100 range
    valley_idx = np.argmin(low_range)  # Find the minimum
    
    # The valley is usually between 20-50
    optimal_threshold = valley_idx
    
    print("=" * 60)
    print("📊 CORRECT SATURATION ANALYSIS")
    print("=" * 60)
    print(f"Valley found at: {optimal_threshold}")
    print(f"This is the true separation point between background and leaf")
    print("=" * 60)
    
    # Visualize with correct threshold
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histogram
    ax1.hist(s.flatten(), bins=256, range=(0, 256), color='green', alpha=0.7)
    ax1.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal: {optimal_threshold}')
    ax1.axvline(x=85, color='blue', linestyle=':', linewidth=2,
                label=f'Current: 85')
    ax1.set_title('Saturation Histogram')
    ax1.set_xlabel('Saturation Value')
    ax1.set_ylabel('Pixel Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mask with optimal threshold
    mask_optimal = pcv.threshold.binary(gray_img=s, threshold=optimal_threshold, 
                                        object_type='light')
    masked_optimal = pcv.apply_mask(img, mask_optimal, 'black')
    ax2.imshow(masked_optimal)
    ax2.set_title(f'Optimal Threshold ({optimal_threshold})')
    ax2.axis('off')
    
    # Mask with current threshold (85)
    mask_current = pcv.threshold.binary(gray_img=s, threshold=85, object_type='light')
    masked_current = pcv.apply_mask(img, mask_current, 'black')
    ax3.imshow(masked_current)
    ax3.set_title(f'Current Threshold (85)')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return optimal_threshold


def parse_args(): # Parse command-line arguments (-src, -dst, -mask)
    parser = argparse.ArgumentParser(description='Leaf image transformation')

    parser.add_argument(
        '-src',
        '--source',
        type=Path,
        help='Path to the source directory containing leaf images'
    )

    parser.add_argument(
        '-dst',
        '--destination',
        type=Path,
        help='Path to the destination directory where transformed images will be saved'
    )

    parser.add_argument(
        '-mask',
        action='store_true',
        help='Generates masks'
    )

    parser.add_argument(
        '-blur',
        action='store_true',
        help='Applies Gaussian blur to the images'
    )

    parser.add_argument(
        '-original',
        action='store_true',
        help='Uses original images without any transformation'
    )

    parser.add_argument(
        '-analyze',
        action='store_true',
        help='Analyzes the images for size and shape'
    )

    parser.add_argument(
        '-pseudolandmarks',
        action='store_true',
        help='Generates pseudolandmarks for the images'
    )

    parser.add_argument(
        '-roi',
        action='store_true',
        help='Finds regions of interest (ROI) in the images'
    )

    parser.add_argument(
        'image_files',
        type=Path,
        nargs='?',
        help='Paths to the image file',
    )

    return parser


def show_image(image, title=None, figsize=(6, 6)):
    """
    Display an image using Matplotlib.

    Parameters:
        image   : NumPy array (returned by PlantCV/OpenCV)
        title   : Window title
        figsize : Figure size (width, height)
    """

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    # plt.axis("off")
    plt.show()


def visualize_colorspace(img):
    return pcv.visualize.colorspaces(rgb_img=img)

def visualize_histogram(img, title='Histogram'):
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


class ImageProcessor:
    def __init__(self, img, dst, specific):
        self.pathname = img
        self.img, self.path, self.filename = pcv.readimage(filename=img)
        self.dst = dst
        self.specific = specific
        self.mask = None

    def gaussian_blur(self):
        blur = pcv.gaussian_blur(
            img=self.img,
            ksize=(5, 5),
            sigma_x=0, 
            sigma_y=None
        )

        s = pcv.rgb2gray_hsv (
            rgb_img=blur,
            channel="s"
        )
        s_treshold = pcv.threshold.binary (
            gray_img=s,
            threshold=85,
            object_type="light"
        )
        return s_treshold

    def create_mask(self):

        # Reduce noise
        blur = pcv.gaussian_blur(
            img=self.img,
            ksize=(5, 5)
        )

        # Convert to HSV (Saturation channel)
        s = pcv.rgb2gray_hsv(
            rgb_img=blur,
            channel="s"
        )

        # Threshold
        s_treshold = pcv.threshold.binary (
            gray_img=s,
            threshold=60,
            object_type="light"
        )

        # Apply mask
        mask = pcv.apply_mask(
            img=self.img,
            mask=s_treshold,
            mask_color="white"
        )
        self.mask = mask

        return mask

    def find_roi_objects(self):
        # Step 1: Convert to LAB 'a' channel
        gray_mask = pcv.rgb2gray_lab(
            rgb_img=self.img,
            channel="a"
        )
        # Step 2: Threshold
        binary_mask = pcv.threshold.binary(
            gray_img=gray_mask,
            threshold=120,
            object_type="dark"
        )
        # Step 3: Create ROI (full image)
        roi = pcv.roi.rectangle(
            img=self.img,           # ← Use original img for display
            x=0,
            y=0,
            h=self.img.shape[0],
            w=self.img.shape[1]
        )
        # Step 4: Filter mask by ROI
        # roi.filter does NOT take roi_type — it keeps objects inside the ROI by default
        filtered_mask = pcv.roi.filter(
            mask=binary_mask,
            roi=roi
            # NO roi_type parameter here!
        )
        return filtered_mask

    def analyze_object(self):
        if self.specific == 'analyze':
            pcv.analyze.size(
                img=self.img,
                labeled_mask=self.mask,
                n_labels=1
            )

    def pseudolandmarks(self):

        if self.specific == 'pseudolandmarks':
            top, bottom, center_v, left, right = pcv.homology.x_axis_pseudolandmarks(
                img=self.img,
                mask=self.mask
            )
        return top, bottom, left, right


    def apply_specifics(self):
        self.gaussian_blur()
        self.create_mask()
        self.find_roi_objects()
        self.analyze_object()


def get_specifics(args):
    specifics = None
    if args.original:
        specifics = 'original'
    elif args.mask:
        specifics = 'mask'
    elif args.source and args.destination:
        specifics = 'process'
    elif args.image_files:
        specifics = 'analyze'
    return specifics



def main():
    
    args = parse_args().parse_args()
    specs = get_specifics(args)

    if args.image_files: # If image files are provided
        img, path, filename = pcv.readimage(filename=args.image_files)
        # pcv.params.debug = "plot" 
        t = ImageProcessor(img=args.image_files, dst=None, specific=specs)
        apply_mask = t.find_roi_objects()
        show_image(image=apply_mask, title='roi_objects')


    if args.source and args.destination: # If source and destination directories are provided
        if not args.source.exists():
            print(f"Source directory {args.source} does not exist.")
            sys.exit(1)
        if not args.destination.exists():
            print(f"Destination directory {args.destination} does not exist. Creating it.")
            args.destination.mkdir(parents=True, exist_ok=True)
        
        for img_path in args.source.glob('*.jpg'):  # Assuming images are in .jpg format
            # vis = visualize_colorspace(img=img)
            t = ImageProcessor(img=img_path, dst=args.destination, specific=specs)
            t.apply_specifics()
            output_path = args.destination / filename
            pcv.print_image(apply_mask, output_path)
            print(f"Processed and saved: {output_path}")


if __name__ == '__main__':
    print(pcv.__version__)
    main()