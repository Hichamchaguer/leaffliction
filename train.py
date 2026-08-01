#!/usr/bin/env python3
"""
train.py — Part 4.1 of the Leaffliction project.

Trains a CNN (MobileNetV2 transfer learning) on the balanced,
augmented leaf-disease dataset produced by Augmentation.py, then
saves the trained model + class labels + dataset into a .zip file,
plus a signature.txt containing its sha1 hash.

Usage:
    python3 train.py ./augmented_directory
    python3 train.py ./augmented_directory
    --epochs-frozen 15 --output leaffliction.zip
"""

import sys
import json
import zipfile
import argparse
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — CLI arguments
# ═══════════════════════════════════════════════════════════════════════
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a leaf-disease classifier on an augmented "
        "dataset using MobileNetV2 transfer learning."
    )
    parser.add_argument(
        "directory", type=str,
        help="Path to the augmented/balanced dataset directory."
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Square image size in pixels (default: 256)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size (default: 32)."
    )
    parser.add_argument(
        "--epochs-frozen", type=int, default=10,
        help="Epochs to train with the base model frozen (default: 10)."
    )
    parser.add_argument(
        "--epochs-finetune", type=int, default=10,
        help="Epochs to fine-tune unfrozen top layers (default: 10)."
    )
    parser.add_argument(
        "--output", type=str, default="leaffliction.zip",
        help="Output zip filename (default: leaffliction.zip)."
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory to save model artifacts (default: models)."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.directory)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"ERROR: '{dataset_dir}' is not a valid directory.")
        sys.exit(1)

    print(f"Dataset directory : {dataset_dir}")
    print(f"Image size        : {args.image_size}x{args.image_size}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Frozen epochs     : {args.epochs_frozen}")
    print(f"Fine-tune epochs  : {args.epochs_finetune}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════
    # Step 2 — Load the dataset (80/20 split, matching seed)
    # ═════════════════════════════════════════════════════════════════
    IMG_SIZE = (args.image_size, args.image_size)
    SEED = 42

    print("\n[Step 2] Loading dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    assert class_names == val_ds.class_names, (
        "Train/val class_names mismatch — check that both calls "
        "use the same seed."
    )

    print(f"  Classes found      : {num_classes}")
    print(f"  Class names        : {class_names}")
    print(f"  Training batches   : {len(train_ds)}")
    print(f"  Validation batches : {len(val_ds)}")

    # ═════════════════════════════════════════════════════════════════
    # Step 3 — Save class names to disk
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 3] Saving class labels...")
    labels_txt_path = models_dir / "labels.txt"
    labels_json_path = models_dir / "class_names.json"

    with open(labels_txt_path, "w") as f:
        f.write("\n".join(class_names))

    with open(labels_json_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"  Saved -> {labels_txt_path}")
    print(f"  Saved -> {labels_json_path}")

    # ═════════════════════════════════════════════════════════════════
    # Step 4 — Preprocessing pipeline
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 4] Building preprocessing pipeline...")
    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(image, label):
        image = preprocess_input(image)
        return image, label

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    sample_images, _ = next(iter(train_ds))
    print(f"  Preprocessed value range : "
          f"[{sample_images.numpy().min():.3f}, "
          f"{sample_images.numpy().max():.3f}]")
    print("  (MobileNetV2 preprocess_input expects roughly [-1, 1])")

    # ═════════════════════════════════════════════════════════════════
    # Step 5 — Build the model with transfer learning
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 5] Building model (MobileNetV2 transfer learning)...")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(args.image_size, args.image_size, 3),
    )
    base_model.trainable = False  # freeze for initial training

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    trainable_params = np.sum(
        [np.prod(v.shape) for v in model.trainable_weights]
    )
    non_trainable_params = np.sum(
        [np.prod(v.shape) for v in model.non_trainable_weights]
    )
    print(f"  Trainable params     : {trainable_params:,}")
    print(f"  Non-trainable params : {non_trainable_params:,}")
    assert non_trainable_params > trainable_params, (
        "Base model does not appear to be frozen — check "
        "base_model.trainable is set to False."
    )

    # ═════════════════════════════════════════════════════════════════
    # Step 6 — First training run (frozen base)
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 6] Training classification head (frozen base)...")

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
    )

    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_frozen,
        callbacks=[early_stop],
    )

    frozen_final_acc = history_frozen.history["accuracy"][-1]
    frozen_final_val_acc = history_frozen.history["val_accuracy"][-1]
    print(f"  End of frozen training : "
          f"accuracy={frozen_final_acc:.3f}, "
          f"val_accuracy={frozen_final_val_acc:.3f}")

    # ═════════════════════════════════════════════════════════════════
    # Step 7 — Fine-tuning (unfreeze top layers)
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 7] Fine-tuning top layers of the base model...")

    base_model.trainable = True
    FINE_TUNE_AT = len(base_model.layers) - 30  # unfreeze last 30 layers
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = models_dir / "best_model.h5"
    checkpoint = ModelCheckpoint(
        str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
    )
    early_stop_ft = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=[checkpoint, early_stop_ft],
    )

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"  Post-fine-tune validation accuracy : {val_acc:.3f}")
    if val_acc < frozen_final_val_acc:
        print("  WARNING: fine-tuning did not improve validation "
              "accuracy — consider a lower learning rate or fewer "
              "unfrozen layers.")

    # ═════════════════════════════════════════════════════════════════
    # Step 8 — Evaluation report
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 8] Generating classification report...")

    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3
    )
    print(report)

    report_path = models_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved -> {report_path}")

    overall_accuracy = float(val_acc)
    if overall_accuracy < 0.90:
        print(f"  WARNING: validation accuracy {overall_accuracy:.1%} "
              f"is below the required 90% threshold.")
    else:
        print(f"  Validation accuracy {overall_accuracy:.1%} meets "
              f"the 90% requirement.")

    # ═════════════════════════════════════════════════════════════════
    # Step 9 — Save model and package everything into a .zip
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 9] Saving model and creating zip archive...")

    model_path = models_dir / "model.h5"
    model.save(str(model_path))
    print(f"  Saved -> {model_path}")

    zip_path = Path(args.output)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname="model.h5")
        zf.write(labels_txt_path, arcname="labels.txt")
        zf.write(labels_json_path, arcname="class_names.json")
        zf.write(report_path, arcname="classification_report.txt")

        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                arcname = Path("dataset") / file_path.relative_to(
                    dataset_dir
                )
                zf.write(file_path, arcname=str(arcname))

    zip_size_mb = zip_path.stat().st_size / 1e6
    print(f"  Created -> {zip_path} ({zip_size_mb:.1f} MB)")

    # ═════════════════════════════════════════════════════════════════
    # Step 10 — Generate signature.txt (sha1 hash of the zip)
    # ═════════════════════════════════════════════════════════════════
    print("\n[Step 10] Generating signature.txt...")

    sha1 = hashlib.sha1()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    digest = sha1.hexdigest()

    signature_path = Path("signature.txt")
    with open(signature_path, "w") as f:
        f.write(f"{digest}  {zip_path.name}\n")

    print(f"  {digest}  {zip_path.name}")
    print(f"  Saved -> {signature_path}")

    print("\n\u2714 Training complete.")


if __name__ == "__main__":
    main()
