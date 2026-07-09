import os
import cv2
import numpy as np


def save_img(img_path: str, src: str, img: np.ndarray, label: str) -> None:
    """Save an image with a modified filename.

    Args:
        img_path (str): Directory path where the image will be saved.
        src (str): Original image filename.
        img (np.ndarray): Image array to save.
        label (str): Suffix to add to the filename before extension.
    """
    name = src.split('.')
    new_name = img_path + '/' + name[0] + label + '.' + name[1]
    cv2.imwrite(new_name, img)


def flip(img_path: str, src: str) -> None:
    """Flip an image vertically and save it.

    Args:
        img_path (str): Directory containing the image.
        src (str): Image filename.
    """
    img = cv2.imread(os.path.join(img_path, src))
    flip_img = cv2.flip(img, 0)
    save_img(img_path, src, flip_img, "_Flip")


def rotate(img_path: str, src: str) -> None:
    """Rotate an image by 45 degrees and save it.

    Args:
        img_path (str): Directory containing the image.
        src (str): Image filename.
    """
    img = cv2.imread(os.path.join(img_path, src))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (n_w / 2) - center[0]
    matrix[1, 2] += (n_h / 2) - center[1]
    rotated_img = cv2.warpAffine(img, matrix, (n_w, n_h))
    save_img(img_path, src, rotated_img, "_Rotate")


def skew(img_path: str, src: str) -> None:
    """Apply a perspective skew to the image and save it.

    Args:
        img_path (str): Directory containing the image.
        src (str): Image filename.
    """
    img = cv2.imread(os.path.join(img_path, src))
    h, w = img.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    skew_amount = int(w * 0.2)
    dst_points = np.float32([
                            [skew_amount, 0],
                            [w - skew_amount, 0],
                            [0, h],
                            [w, h]
                            ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    skewed_img = cv2.warpPerspective(img, M, (w, h))
    save_img(img_path, src, skewed_img, "_Skew")


def shear(img_path: str, src: str) -> None:
    """Apply a shear transformation to the image and save it.

    Args:
        img_path (str): Directory containing the image.
        src (str): Image filename.
    """
    img = cv2.imread(os.path.join(img_path, src))
    h, w = img.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.float32([
                            [100, 0],
                            [w + 100, 0],
                            [0, h],
                            [w, h]
                            ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    shear_img = cv2.warpPerspective(img, M, (w + 100, h))
    save_img(img_path, src, shear_img, "_Shear")


def crop(img_path: str, src: str) -> None:
    """Crop the center region, zoom in 2x, and resize back to original size.

    Args:
        img_path (str): Directory containing the image.
        src (str): Image filename.
    """
    img = cv2.imread(os.path.join(img_path, src))
    h, w = img.shape[:2]

    center_x, center_h = w // 2, h // 2
    zoom = 2

    crop_x = int(w / zoom)
    crop_h = int(h / zoom)

    start_x = max(0, center_x - crop_x // 2)
    start_h = max(0, center_h - crop_h // 2)
    end_x = min(w, start_x + crop_x)
    end_h = min(h, start_h + crop_h)

    zoomed_img = img[start_h: end_h, start_x: end_x]
    croped_img = cv2.resize(zoomed_img, (w, h), interpolation=cv2.INTER_LINEAR)
    save_img(img_path, src, croped_img, "_Crop")


def distortion(img_path: str, src: str) -> None:
    """Apply a radial lens distortion effect to the image and save it.

    Args:
        img_path (str): Directory containing the image.
        src (str): Image filename.
    """
    img = cv2.imread(os.path.join(img_path, src))
    h, w = img.shape[:2]

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    c_x, c_y = w / 2, h / 2
    stren = 0.00005

    d_x = map_x - c_x
    d_y = map_y - c_y
    r2 = d_x ** 2 + d_y ** 2

    dist_x = c_x + d_x * (1 + stren * r2)
    dist_y = c_y + d_y * (1 + stren * r2)

    distorted_img = cv2.remap(img, dist_x, dist_y, cv2.INTER_LINEAR)
    save_img(img_path, src, distorted_img, "_Distortion")

    # Thos line just to test if the my function works well
    # cv2.imshow('Image', distorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
