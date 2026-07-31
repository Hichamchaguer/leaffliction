import os
import cv2
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from srcs.tools import check_img


def predict_image(image_path):
    """_summary_

    Args:
        data_dir (_type_): _description_
    """
    try:
        # 1. Charger le modèle et les noms de classes
        model = tf.keras.models.load_model('leaf_model.h5')
        with open('classes.txt', 'r') as f:
            class_names = f.read().splitlines()

        # 2. Préparer l'image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)

        # 3. Prédire
        predictions = model.predict(img_array)
        # score = tf.nn.softmax(predictions[0])
        result = class_names[np.argmax(predictions)]

        # 4. Affichage (comme demandé dans le sujet)
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img_rgb)

        plt.subplot(1, 2, 2)
        plt.title(f"Prédiction : {result}")
        plt.imshow(img_resized)
        plt.show()

        print(f"L'image appartient probablement à : {result}")

    except Exception as err:
        print(f"Error exception {err}")
        print(f"line: {err.__traceback__.tb_lineno}")


def main():
    try:
        assert len(sys.argv) == 2, "Usage: python train.py [dossier_images]"
        __dir1 = sys.argv[1]

        if not check_img(__dir1):
            sys.exit(1)
        predict_image(__dir1)

    except Exception as err:
        print(f"Error exception {err}")
        print(f"line: {err.__traceback__.tb_lineno}")


if __name__ == "__main__":
    main()
