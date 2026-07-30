import sys
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.mobilenet_v2 import (  # noqa: E402
    preprocess_input,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — CLI arguments
# ═══════════════════════════════════════════════════════════════════════
def build_parser():
    parser = argparse.ArgumentParser(
        description="Classify a leaf image, or batch-test accuracy "
                     "over a folder of images."
    )
    parser.add_argument(
        "path", type=str,
        help="Path to a single image, or a folder (with --batch-test)."
    )
    parser.add_argument(
        "--model", type=str, default="models/model.h5",
        help="Path to the trained model file (default: models/model.h5)."
    )
    parser.add_argument(
        "--labels", type=str, default="models/labels.txt",
        help="Path to the labels file (default: models/labels.txt)."
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Square image size used during training (default: 256)."
    )
    parser.add_argument(
        "--batch-test", type=int, default=None,
        help="If PATH is a folder, sample this many random images "
             "and report prediction accuracy."
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Load model and labels
# ═══════════════════════════════════════════════════════════════════════
def load_model_and_labels(model_path, labels_path):
    model_path = Path(model_path)
    labels_path = Path(labels_path)

    if not model_path.exists():
        print(f"ERROR: model file '{model_path}' not found. "
              f"Run train.py first.")
        sys.exit(1)
    if not labels_path.exists():
        print(f"ERROR: labels file '{labels_path}' not found. "
              f"Run train.py first.")
        sys.exit(1)

    print(f"Loading model  -> {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    class_names = labels_path.read_text().strip().split("\n")
    print(f"Loading labels -> {labels_path}  ({len(class_names)} classes)")

    return model, class_names


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Preprocess the input image
# ═══════════════════════════════════════════════════════════════════════
def preprocess_image(image_path, target_size):
    """
    Load an image and prepare it exactly the way train.py's dataset
    pipeline did: resize to (target_size, target_size), then apply
    MobileNetV2's preprocess_input, then add the batch dimension.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_bgr = cv2.resize(img_bgr, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    arr = img_rgb.astype(np.float32)
    arr = preprocess_input(arr)          # same normalisation as training
    arr = np.expand_dims(arr, axis=0)    # (1, H, W, 3) — batch of one
    return arr


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Run inference
# ═══════════════════════════════════════════════════════════════════════
def predict_image(model, class_names, image_path, image_size):
    arr = preprocess_image(image_path, image_size)
    predictions = model.predict(arr, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    return predicted_class, confidence, predictions


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Display original + transformed side by side
# ═══════════════════════════════════════════════════════════════════════
def get_transformed_panel(image_path, image_size):
    """
    Reuse a transform from Transformation.py for the right-hand
    display panel. Falls back to a plain Gaussian blur if
    Transformation.py is not importable (e.g. run from a different
    directory), so predict.py still works standalone.
    """
    try:
        from Transformation import ImageTransformation, Options

        opt = Options(str(image_path))
        transformer = ImageTransformation(image_path, dest=None, opt=opt)
        transformer.original()
        transformer.threshold()
        transformer.m_blur()
        blurred = transformer.blur()

        if blurred.ndim == 2:
            transformed_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        else:
            transformed_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        return transformed_rgb, "Gaussian Blur (Transformation.py)"

    except Exception as e:
        print(f"  [display] Transformation.py unavailable ({e}); "
              f"using fallback blur.")
        img_bgr = cv2.imread(str(image_path))
        img_bgr = cv2.resize(img_bgr, (image_size, image_size))
        blurred_bgr = cv2.GaussianBlur(img_bgr, (15, 15), 0)
        transformed_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
        return transformed_rgb, "Gaussian Blur (fallback)"


def display_prediction(image_path, predicted_class, confidence,
                        image_size):
    img_bgr = cv2.imread(str(image_path))
    img_bgr = cv2.resize(img_bgr, (image_size, image_size))
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transformed_rgb, transform_label = get_transformed_panel(
        image_path, image_size
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    ax1.imshow(original_rgb)
    ax1.set_title("Original", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(transformed_rgb)
    ax2.set_title(transform_label, fontsize=12, fontweight="bold")
    ax2.axis("off")

    fig.suptitle(
        f"Class predicted: {predicted_class}  ({confidence:.1f}%)",
        fontsize=14, fontweight="bold", color="green",
    )
    plt.tight_layout()
    plt.show(block=True)


# ═══════════════════════════════════════════════════════════════════════
# Step 6 — Batch accuracy testing (optional)
# ═══════════════════════════════════════════════════════════════════════
def batch_test_accuracy(folder, model, class_names, image_size, n=100):
    folder = Path(folder)
    all_images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix in IMAGE_EXTS
    ]

    if not all_images:
        print(f"ERROR: no images found under '{folder}'.")
        sys.exit(1)

    sample_size = min(n, len(all_images))
    sample = random.sample(all_images, sample_size)

    correct = 0
    per_class_total = {}
    per_class_correct = {}

    print(f"Testing {sample_size} random images from '{folder}'...\n")

    for i, img_path in enumerate(sample, start=1):
        true_class = img_path.parent.name  # folder name = ground truth
        try:
            predicted_class, confidence, _ = predict_image(
                model, class_names, img_path, image_size
            )
        except Exception as e:
            print(f"  [{i}/{sample_size}] SKIPPED ({img_path.name}): {e}")
            continue

        is_correct = predicted_class == true_class
        correct += int(is_correct)

        per_class_total[true_class] = per_class_total.get(true_class, 0) + 1
        if is_correct:
            per_class_correct[true_class] = (
                per_class_correct.get(true_class, 0) + 1
            )

        status = "OK  " if is_correct else "MISS"
        print(f"  [{i}/{sample_size}] {status}  true={true_class:<25} "
              f"pred={predicted_class:<25} ({confidence:.1f}%)")

    accuracy = 100 * correct / sample_size
    print(f"\n{correct}/{sample_size} ({accuracy:.1f}%) predicted "
          f"correctly.\n")

    print("Per-class breakdown:")
    for cls in sorted(per_class_total):
        total = per_class_total[cls]
        right = per_class_correct.get(cls, 0)
        print(f"  {cls:<25} {right}/{total} "
              f"({100 * right / total:.1f}%)")

    return accuracy


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = build_parser()
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: '{path}' does not exist.")
        sys.exit(1)

    model, class_names = load_model_and_labels(args.model, args.labels)

    if path.is_dir():
        if args.batch_test is None:
            print("ERROR: PATH is a folder — pass --batch-test N to "
                  "run a batch accuracy test, or point PATH at a "
                  "single image file instead.")
            sys.exit(1)
        batch_test_accuracy(
            path, model, class_names, args.image_size, n=args.batch_test
        )

    elif path.is_file():
        predicted_class, confidence, _ = predict_image(
            model, class_names, path, args.image_size
        )
        print(f"Class predicted : {predicted_class}")
        print(f"Confidence      : {confidence:.1f}%")
        display_prediction(path, predicted_class, confidence,
                            args.image_size)

    else:
        print(f"ERROR: '{path}' is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()