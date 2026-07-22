import sys
import pathlib
import random
import cv2
import albumentations as alb


TRANSFORMS = {
    "flip": alb.HorizontalFlip(p=1.0),
    "rotate": alb.Rotate(limit=30,p=1.0),
    "skew": alb.Perspective(scale=(0.03, 0.08),p=1.0),
    "shear": alb.Affine(shear=(-10, 10),p=1.0),
    "crop": alb.RandomResizedCrop(size=(256, 256),scale=(0.85, 1.0),ratio=(0.9, 1.1),p=1.0),
    "distortion": alb.OpticalDistortion(distort_limit=0.05,p=1.0)
}

def apply_augmentation(image):

    augmented_name = random.choice(list(TRANSFORMS.keys())) # choose a random transforms :
    transform = TRANSFORMS[augmented_name]

    augmented = transform(image=image)["image"]
    return augmented, augmented_name

def check_count(folder, label):
    return len(list(folder.glob(f"{label}/*")))

def main():
    if len(sys.argv) != 2:
        print("Usage: python augmentation.py <input_file>")
        sys.exit(1)

    input_folder = sys.argv[1]

    # convert input_folder into pathlib         

    data_dir = pathlib.Path(input_folder)
    black_rot = list(data_dir.glob("Apple_Black_rot/*"))

    output_dir = pathlib.Path("augmented_directory")
    data_img_dict = {}

    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            data_img_dict[class_dir.name] =list(class_dir.glob("*"))

    max_len = max(len(images) for images in data_img_dict.values())

    for label, images in data_img_dict.items():

        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        current = len(images)
        missing = max_len - current
        # print("=" * 50)
        # print(f"Class   : {label}")
        # print(f"Current : {current}")
        # print(f"Need    : {missing}")

        
        if check_count(output_dir, label) >= max_len:
            print(f"Class {label} is already balanced in the output directory.")
            continue
        for img_path in images:
            img = cv2.imread(str(img_path)) # read the images
            if img is not None:
                output_path = class_dir / img_path.name
                cv2.imwrite(str(output_path), img)

        if missing == 0:
            print(f"Class {label} is already balanced.")
            continue
        for i in range(missing):

            # Pick one original image randomly
            img = random.choice(images)
            image = cv2.imread(str(img)) # read the images
            # convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Apply augmentation
            augmented, augmented_name = apply_augmentation(image)
            # reconvert RGB to BGR
            augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            # Save the augmented image
            output_path = (
                class_dir /
                f"{augmented_name}_{i}_{img.name}")

            cv2.imwrite(str(output_path), augmented)

        print(f"Number of images in {label}: {len(list(output_dir.glob(f'{label}/*')))}")
        #     print(
        #         f"{i+1:4d} -> {img.name}" # aumentation part 
        #     )
        # print()

if __name__ == "__main__":
    main()