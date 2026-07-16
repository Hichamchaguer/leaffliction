import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import sys
import warnings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] \
        in %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class Augmentation:
    '''
    This class is used to augment the images in the dataset.
    Nine (9) different types of augmentation are used:
    1. Translation
    2. Shear
    3. Flip
    4. Rotate
    5. Crop
    6. Skew
    7. Distortion
    8. Blur
    9. Contrast
    '''

    def __init__(self):
        pass

    def translation(self, img: np.ndarray) -> np.ndarray:
        '''
        Translate the image by a random number of pixels
        in the x and y direction.

        Args:
            img (np.ndarray): The image to be translated.

        Returns:
            translated_img (np.ndarray): The translated image.
        '''
        rows, cols, dim = img.shape
        M = np.float32([[1, 0, np.random.randint(-100, 100)],
                        [0, 1, np.random.randint(-100, 100)],
                        [0, 0, 1]])
        translated_img = cv2.warpPerspective(img, M, (cols, rows))
        translated_img = translated_img[50:150, 50:150]
        return translated_img

    def shear(self, img, axis=0) -> np.ndarray:
        '''
        Shear the image by a random number of pixels
        in the x and y direction.

        Args:
            img (np.ndarray): The image to be sheared.
            axis (int, optional): The axis along which the image
                                    is to be sheared. Defaults to 0.

        Returns:
            sheared_img (np.ndarray): The sheared image.
        '''
        rows, cols, dim = img.shape
        if axis == 0:
            M = np.float32([[1, np.random.uniform(0.1, 0.2), 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        else:
            M = np.float32([[1, 0, 0],
                            [np.random.uniform(0.1, 0.2), 1, 0],
                            [0, 0, 1]])
        sheared_img = cv2.warpPerspective(img, M, (cols, rows))
        sheared_img = sheared_img[50:150, 50:150]
        return sheared_img

    def flip(self, img: np.ndarray, axis: int = 0) -> np.ndarray:
        '''
        Flip the image along the x or y axis.

        Args:
            img (np.ndarray): The image to be flipped.
            axis (int, optional): The axis along which the image
                                    is to be flipped. Defaults to 0.

        Returns:
            reflected_img (np.ndarray): The flipped image.
        '''
        rows, cols, dim = img.shape
        if axis == 0:
            M = np.float32([[-1, 0, cols],
                            [0, 1, 0],
                            [0, 0, 1]])
        else:
            M = np.float32([[1, 0, 0],
                            [0, -1, rows],
                            [0, 0, 1]])
        reflected_img = cv2.warpPerspective(img, M, (cols, rows))
        return reflected_img

    def rotate(self, img: np.ndarray) -> np.ndarray:
        '''
        Rotate the image by a random angle.

        Args:
            img (np.ndarray): The image to be rotated.

        Returns:
            rotated_img (np.ndarray): The rotated image.
        '''
        rows, cols, dim = img.shape
        M = cv2.getRotationMatrix2D(
            (cols/2, rows/2), np.random.randint(-180, 180), 1)
        rotated_img = cv2.warpAffine(img, M, (cols, rows))
        rotated_img = rotated_img[50:150, 50:150]
        return rotated_img

    def crop(self, img: np.ndarray) -> np.ndarray:
        '''
        Crop the image by a random number of pixels in the x and y direction.

        Args:
            img (np.ndarray): The image to be cropped.

        Returns:
            cropped_img (np.ndarray): The cropped image.
        '''
        rows, cols, dim = img.shape
        x = np.random.randint(0, cols-100)
        y = np.random.randint(0, rows-100)
        cropped_img = img[y:y+100, x:x+100]
        return cropped_img

    def skew(self, img: np.ndarray) -> np.ndarray:
        '''
        Skew the image by a random number of pixels in the x and y direction.

        Args:
            img (np.ndarray): The image to be skewed.

        Returns:
            skewed_img (np.ndarray): The skewed image.
        '''
        rows, cols, dim = img.shape
        pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
        pts2 = np.float32(
            [[np.random.randint(0, 30), np.random.randint(0, 30)],
             [cols-np.random.randint(0, 30), np.random.randint(0, 30)],
             [np.random.randint(0, 30), rows-np.random.randint(0, 30)]]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        skewed_img = cv2.warpAffine(img, M, (cols, rows))
        skewed_img = skewed_img[50:150, 50:150]
        return skewed_img

    def distortion(self, img: np.ndarray) -> np.ndarray:
        '''
        Distort the image by a random number of pixels
        in the x and y direction.

        Args:
            img (np.ndarray): The image to be distorted.

        Returns:
            distorted_img (np.ndarray): The distorted image.
        '''
        rows, cols, dim = img.shape
        pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        pts2 = np.float32(
            [[np.random.randint(0, 30), np.random.randint(0, 30)],
             [cols-np.random.randint(0, 30), np.random.randint(0, 30)],
             [np.random.randint(0, 30), rows-np.random.randint(0, 30)],
             [cols-np.random.randint(0, 30), rows-np.random.randint(0, 30)]]
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        distorted_img = cv2.warpPerspective(img, M, (cols, rows))
        distorted_img = distorted_img[50:150, 50:150]
        return distorted_img

    def blur(self, img: np.ndarray) -> np.ndarray:
        '''
        Blur the image.

        Args:
            img (np.ndarray): The image to be blurred.

        Returns:
            blurred_img (np.ndarray): The blurred image.
        '''
        blurred_img = cv2.blur(img, (5, 5), 0)
        return blurred_img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        '''
        Change the contrast of the image.

        Args:
            img (np.ndarray): The image whose contrast is to be changed.

        Returns:
            contrast_img (np.ndarray): The image with changed contrast.
        '''
        contrast_img = cv2.addWeighted(
            img, 2, np.zeros(img.shape, img.dtype), 0, -50)
        return contrast_img


def augment(
    image: Path,
    save_path: Path,
    len_largest_directory: int,
    aug: Augmentation, augmentation_type: str
) -> None:
    logger.debug(
        f'Number of images in {image.parent.stem}: \
            {len(list(save_path.iterdir()))}')
    logger.debug(
        f'Number of images in largest directory: {len_largest_directory}')
    aug_img = getattr(aug, augmentation_type)(cv2.imread(str(image)))
    aug_img = cv2.resize(aug_img, (256, 256))
    plt.imsave(
        Path(
            save_path,
            f'{image.stem}_{augmentation_type.title()}.JPG'),
        aug_img
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment image(s) in the dataset. Augmentation applied: \
                        Translation, Flip, Rotate, Blur, Crop, Distortion. \
                        Images are saved in the "augmented_directory" folder.'
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to the image/directory to be augmented.'
    )
    args = parser.parse_args()

    try:
        if not os.path.isdir(args.path) and not os.path.isfile(args.path):
            raise FileNotFoundError('Path not found.')
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    if os.path.isdir(args.path):
        images = Path(args.path).glob('**/*.JPG')
        for image in tqdm(images, desc=f'Copying images from {args.path} \
                to augmented_directory', total=len(list(Path(args.path).glob('**/*.JPG')))):
            save_path = Path(
                'data/images/augmented_directory/',
                image.parent.stem)
            os.makedirs(save_path, exist_ok=True)
            img = cv2.imread(str(image))  # BGR format
            cv2.imwrite(str(Path(save_path, image.name)), img)

        largest_directory = max(
            Path(
                'data/images/augmented_directory/').iterdir(),
            key=lambda x: len(list(x.iterdir()))
        )
        len_largest_directory = len(list(largest_directory.iterdir()))
        logger.debug(f'Largest directory: {largest_directory}')
        logger.debug(
            f'Number of images in largest directory: {len_largest_directory}')

        for image in tqdm(
            Path(args.path).iterdir(),
            desc=f'Augmenting images from {args.path}',
            total=len(os.listdir(args.path))
        ):
            images = image.glob('**/*.JPG')
            for image in images:
                aug = Augmentation()
                img = cv2.imread(str(image))
                save_path = Path(
                    'data/images/augmented_directory/',
                    image.parent.stem)
                os.makedirs(save_path, exist_ok=True)
                plt.imsave(Path(save_path, image.name), img)

                # flip images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path,
                            len_largest_directory, aug, 'flip')
                # rotate images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path,
                            len_largest_directory, aug, 'rotate')
                # blur images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path,
                            len_largest_directory, aug, 'blur')
                # crop images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path,
                            len_largest_directory, aug, 'crop')
                # distort images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path, len_largest_directory,
                            aug, 'distortion')
                # shear images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path,
                            len_largest_directory, aug, 'shear')
                # skew images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path,
                            len_largest_directory, aug, 'skew')
                # contrast images
                if len(list(save_path.iterdir())) < len_largest_directory:
                    augment(image, save_path,
                            len_largest_directory, aug, 'contrast')
    else:
        aug = Augmentation()
        img = cv2.imread(args.path)
        img_name = Path(args.path).stem

        translated_img = aug.translation(img)
        cv2.imwrite(img_name + '_Translation.JPG', translated_img)

        flipped_img = aug.flip(img, axis=np.random.randint(0, 2))
        cv2.imwrite(img_name + '_Flip.JPG', flipped_img)

        rotated_img = aug.rotate(img)
        cv2.imwrite(img_name + '_Rotate.JPG', rotated_img)

        blurred_img = aug.blur(img)
        cv2.imwrite(img_name + '_Blur.JPG', blurred_img)

        cropped_img = aug.crop(img)
        cv2.imwrite(img_name + '_Crop.JPG', cropped_img)

        contrast_img = aug.contrast(img)
        cv2.imwrite(img_name + '_Contrast.JPG', contrast_img)


        plt.figure(figsize=(20, 20))
        plt.subplot(1, 7, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 7, 2)
        plt.imshow(translated_img)
        plt.title('Translated Image')
        plt.axis('off')
        plt.subplot(1, 7, 3)
        plt.imshow(flipped_img)
        plt.title('Flipped Image')
        plt.axis('off')
        plt.subplot(1, 7, 4)
        plt.imshow(rotated_img)
        plt.title('Rotated Image')
        plt.axis('off')
        plt.subplot(1, 7, 5)
        plt.imshow(blurred_img)
        plt.title('Blurred Image')
        plt.axis('off')
        plt.subplot(1, 7, 6)
        plt.imshow(cropped_img)
        plt.title('Cropped Image')
        plt.axis('off')
        plt.subplot(1, 7, 7)
        plt.imshow(contrast_img)
        plt.title('Contrasted Image')
        plt.axis('off')
        plt.show()