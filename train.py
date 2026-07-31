import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models


def train_model(data_dir):
    """_summary_

    Args:
        data_dir (_type_): _description_
    """
    try:
        # 1. Chargement et division des données
        img_height, img_width = 224, 224
        batch_size = 32

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        class_names = train_ds.class_names
        print(f"{'-' * 50 }\nClasses found : {class_names}")

        # 2. Création du modèle (Transfer Learning avec MobileNetV2)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
            )
        base_model.trainable = False

        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(224, 224, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(len(class_names), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # 3. Entraînement
        epochs = 10
        model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        # 4. Sauvegarde
        model.save('leaf_model.h5')
        # with open('classes.txt', 'w') as f:
        #     f.write('\n'.join(class_names))

        print("Entraînement terminé et modèle sauvegardé.")
    except Exception as err:
        print(f"Error exception {err}")
        print(f"line: {err.__traceback__.tb_lineno}")


def main():
    try:
        assert len(sys.argv) == 2, "Usage: python train.py [dossier_images]"
        __dir1 = sys.argv[1]
        if not os.path.isdir(__dir1):
            print("This directory doesn't existe")
            sys.exit(1)
        train_model(__dir1)

    except Exception as err:
        print(f"Error exception {err}")
        print(f"line: {err.__traceback__.tb_lineno}")


if __name__ == "__main__":
    main()
