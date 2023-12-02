import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
from torchvision import transforms


def resize_and_save_images(source_folder, destination_folder, target_size):
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        image = cv2.imread(source_path)

        if image is not None:
            resized_image = cv2.resize(image, target_size)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(destination_path, resized_image)
            

if __name__ == "__main__":
    source_folder_path = 'Assign02/Facades/A/train'
    destination_folder_path = 'Assign03/train_images'

    target_size = (64, 64)

    resize_and_save_images(source_folder_path, destination_folder_path, target_size)
