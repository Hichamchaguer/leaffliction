import argparse
from pathlib import Path
import sys
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt
import numpy as np
import cv2


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


def parse_args():
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
        self.binary_mask = None
        self.mask = None
        self._mask = None
        self.upper_green = np.array([90, 255, 255])  # HSV upper bound for green
        self.lower_green = np.array([25, 40, 40])    # HSV lower bound for green
        self.upper_brown = np.array([25, 255, 2500])  # HSV upper bound for brown
        self.lower_brown = np.array([10, 40, 40])    # HSV lower bound for brown
        self.BLUE_RGB = (0, 0, 255)  # RGB color for blue



    def original(self):
        return self.img

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
        blur = pcv.gaussian_blur (
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
            threshold=90,
            object_type="light"
        )
        self.binary_mask = s_treshold
        # Apply mask
        mask = pcv.apply_mask(
            img=self.img,
            mask=s_treshold,
            mask_color="white"
        )
        self.mask = mask

        return mask

    def _roi_objects_contour(self, mask_img):
        """
        Step IV.4: ROI Objects - Green overlay follows exact leaf contour
        Blue frame = convex hull or bounding contour around the leaf
        """
        # Ensure binary mask
        if len(mask_img.shape) == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self.img
        
        # Keep largest contour (the leaf)
        leaf_contour = max(contours, key=cv2.contourArea)
        
        # Create output
        roi_img = self.img.copy()
        
        # === GREEN OVERLAY: Exact leaf shape ===
        # Create green mask overlay
        green_overlay = np.zeros_like(self.img)
        green_overlay[:] = (0, 255, 0)  # BGR green
        
        # Apply only within contour
        green_mask = np.zeros_like(binary_mask)
        cv2.drawContours(green_mask, [leaf_contour], -1, 255, thickness=cv2.FILLED)
        
        # Bitwise AND to get green only on leaf
        green_leaf = cv2.bitwise_and(green_overlay, green_overlay, mask=green_mask)
        
        # Blend with original (transparency)
        alpha = 0.35
        roi_img = cv2.addWeighted(roi_img, 1.0, green_leaf, alpha, 0)
        
        # === BLUE FRAME: Convex hull around leaf (not rectangle!) ===
        # This creates a tight polygon around the leaf, not a box
        hull = cv2.convexHull(leaf_contour)
        cv2.drawContours(roi_img, [hull], -1, (255, 0, 0), thickness=4)  # Blue
        
        return roi_img

# === COMPLETE PIPELINE ===
    def roi_objects(self):

        original = cv2.imread(self.pathname)
        # Step 1: Gaussian blur
        blurred = cv2.GaussianBlur(original, (5, 5), 0)
        
        # Step 2: Create mask (HSV thresholding)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Step 3: ROI Objects - Choose your method:
        
        # Method A: Convex hull (recommended, matches your figure)
        roi_result = self._roi_objects_contour(mask)
        
        return roi_result

    def analyze_object(self):

        analyse = pcv.analyze.size(
            img=self.img,
            labeled_mask=self.binary_mask,
            n_labels=1
        )
        return analyse

    def pseudolandmarks(self):

        if self.binary_mask is None:
            self.create_mask()


        labeled_mask, n_labels = pcv.create_labels(
            mask=self.binary_mask
        )


        vis = self.img.copy()


        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=self.img,
            mask=self.binary_mask
        )


        groups = [
            (top, (0,0,255)),
            (bottom, (255,0,0)),
            (center_v, (0,255,0))
        ]


        for pts, colour in groups:

            for pt in pts:

                x = int(pt[0][0])
                y = int(pt[0][1])


                cv2.circle(
                    vis,
                    (x,y),
                    4,
                    colour,
                    -1
                )


        return cv2.cvtColor(
            vis,
            cv2.COLOR_BGR2RGB
        )


    def apply_specifics(self):
        self.original()
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
        t = ImageProcessor(img=args.image_files, dst=None, specific=specs)
        t.create_mask()
        apply_mask = t.pseudolandmarks()
        show_image(image=apply_mask, title='landmarks')


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