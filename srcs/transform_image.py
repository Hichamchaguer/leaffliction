import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

matplotlib.use("TkAgg")


class ImageTransformation:
    """_summary_
    """

    def __init__(self, src):
        self.src = src
        self.img, self.path, self.filename = pcv.readimage(filename=self.src)
        self.original = None
        self.blur = None
        self.clean_mask = None
        self.img_mask = None
        self.img_roi = None
        self.img_shape = None
        self.img_pseudo = None
        self.dst = None
        self.transformations = []

    def _original_img(self):
        """_summary_
        """
        try:
            self.original = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def _get_mask(self):
        """_summary_
        """
        try:
            self.blur = pcv.gaussian_blur(self.img, ksize=(5, 5), sigma_x=0)
            lab = pcv.rgb2gray_hsv(rgb_img=self.blur, channel="s")
            mask = pcv.threshold.binary(lab, threshold=85, object_type="light")
            clean_mask = pcv.fill(bin_img=mask, size=50)
            self.clean_mask = pcv.dilate(gray_img=clean_mask, ksize=3, i=1)
        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def _mask_img(self):
        """_summary_
        """
        try:
            self.img_mask = pcv.apply_mask(self.img,
                                           self.clean_mask,
                                           mask_color="white")
        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def _roi_img(self):
        """_summary_
        """
        try:
            roi_img = self.img.copy()
            contours, _ = cv2.findContours(self.clean_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE
                                           )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(roi_img, [largest],
                                 -1, (0, 255, 0),
                                 thickness=cv2.FILLED
                                 )
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(roi_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.img_roi = roi_img
        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def _analyze_img(self):
        """_summary_
        """
        try:
            self.img_shape = pcv.analyze.size(img=self.img,
                                              labeled_mask=self.clean_mask)
        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def _pseudo_img(self):
        """_summary_
        """
        try:
            top, bottom, center = pcv.homology.x_axis_pseudolandmarks(
                img=self.img, mask=self.clean_mask
                )
            pseudo_img = self.img.copy()
            for pt in top:
                cv2.circle(pseudo_img,
                           tuple(pt.astype(int).ravel()),
                           6, (255, 0, 0), -1
                           )
            for pt in bottom:
                cv2.circle(pseudo_img,
                           tuple(pt.astype(int).ravel()),
                           6, (0, 255, 0), -1
                           )
            for pt in center:
                cv2.circle(pseudo_img,
                           tuple(pt.astype(int).ravel()),
                           6, (0, 0, 255), -1
                           )

            self.img_pseudo = pseudo_img

        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def apply_transformations(self):
        """_summary_
        """
        self._original_img()
        self._get_mask()
        self._mask_img()
        self._roi_img()
        self._analyze_img()
        self._pseudo_img()

    def plot_colors(self):
        try:
            fig, axc = plt.subplots(figsize=(10, 8))
            pixels = self.original[self.clean_mask > 0]

            hsv = cv2.cvtColor(self.original, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(self.original, cv2.COLOR_RGB2LAB)

            hsvs = hsv[self.clean_mask > 0]
            labs = lab[self.clean_mask > 0]
            channels = {
                "blue":          pixels[:, 2],
                "blue-yellow":   labs[:, 2],
                "green":         pixels[:, 1],
                "green-magenta": labs[:, 1],
                "hue":           hsvs[:, 0],
                "lightness":     labs[:, 0],
                "red":           pixels[:, 0],
                "saturation":    hsvs[:, 1],
                "value":         hsvs[:, 2],
            }
            colors = {
                "blue": "blue",
                "blue-yellow": "gold",
                "green": "green",
                "green-magenta": "magenta",
                "hue": "purple",
                "lightness": "gray",
                "red": "red",
                "saturation": "cyan",
                "value": "orange",
            }
            for name, data in channels.items():
                counts, bin_edges = np.histogram(
                    data, bins=100,
                    range=(0, 255),
                    density=True
                    )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                axc.plot(bin_centers, counts, color=colors[name], label=name)

            axc.set_title("Figure IV.7: Color histogram")
            axc.set_xlabel("Pixel intensity")
            axc.set_ylabel("Proportion of pixels (%)")
            axc.legend(fontsize=7, loc='upper right')
            axc.set_facecolor("#EAEAF2")
            axc.grid(color="white", zorder=0)
            for spine in axc.spines.values():
                spine.set_visible(False)

        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def transforme_single_image(self):
        """_summary_
        """
        try:
            fig, ax = plt.subplots(3, 2, figsize=(10, 8))
            self.apply_transformations()

            # Display Original image
            ax[0, 0].imshow(self.original)
            ax[0, 0].set_title("Figure IV.1: Original", y=-0.5)

            # Display Blured Image
            ax[0, 1].imshow(self.clean_mask, cmap="gray")
            ax[0, 1].set_title("Figure IV.2: Gaussian blur", y=-0.5)

            # Display Masked Image
            ax[1, 0].imshow(self.img_mask)
            ax[1, 0].set_title("Figure IV.3: Mask", y=-0.5)

            # Display ROI Image
            ax[1, 1].imshow(self.img_roi)
            ax[1, 1].set_title("Figure IV.4: Roi objects", y=-0.5)

            # Display Analyze object Image
            ax[2, 0].imshow(self.img_shape)
            ax[2, 0].set_title("Figure IV.5: Analyze object", y=-0.5)

            # Display Pseudo Image
            ax[2, 1].imshow(self.img_pseudo)
            ax[2, 1].set_title("Figure IV.6: Pseudolandmarks", y=-0.5)

            self.plot_colors()
            plt.tight_layout()
            plt.show()

        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")

    def _safileName(self, signature):
        """_summary_
        """
        try:
            name = self.filename.split(".")
            return name[0] + signature + '.' + name[1]
        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")
        return ''

    def save_transformations(self):
        """_summary_
        """
        try:
            if 'original' in self.transformations:
                pcv.print_image(
                    self.original,
                    os.path.join(self.dst, self._safileName("_Original")))
            if 'blur' in self.transformations:
                pcv.print_image(
                    self.clean_mask,
                    os.path.join(self.dst, self._safileName("_Blur")))
            if 'mask' in self.transformations:
                pcv.print_image(
                    self.img_mask,
                    os.path.join(self.dst, self._safileName("_Mask")))
            if 'roi' in self.transformations:
                pcv.print_image(
                    self.img_roi,
                    os.path.join(self.dst, self._safileName("_Roi")))
            if 'analyze' in self.transformations:
                pcv.print_image(
                    self.img_shape,
                    os.path.join(self.dst, self._safileName("_Analyze")))
            if 'pseudo' in self.transformations:
                pcv.print_image(
                    self.img_pseudo,
                    os.path.join(self.dst, self._safileName("_Pseudo")))
        except Exception as err:
            print(f"Error exception {err}")
            print(f"line: {err.__traceback__.tb_lineno}")
