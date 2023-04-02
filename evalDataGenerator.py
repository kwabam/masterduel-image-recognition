import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from PIL import Image


def create_eval_data(train_folder, eval_folder, num_augmented_images=5):
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    datagen = ImageDataGenerator(
        zoom_range=[.8, 1.2]
    )

    for card_name in os.listdir(train_folder):
        card_folder = os.path.join(train_folder, card_name)
        if os.path.isdir(card_folder):
            for image_file in os.listdir(card_folder):
                image_path = os.path.join(card_folder, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(image_path)
                    image = image.resize((210, 306))
                    image = np.array(image)
                    image = np.expand_dims(image, axis=0)
                    print(f"Augmenting {image_file} for {card_name}...")
                    save_directory = os.path.join(eval_folder, card_name)
                    os.makedirs(save_directory, exist_ok=True)
                    for _ in range(num_augmented_images):
                        for _ in datagen.flow(image,
                                              batch_size=1,
                                              save_to_dir=save_directory,
                                              save_prefix=f"{card_name}-{image_file.split('.')[0]}",
                                              save_format='png'):
                            break


# Set the paths
test = False  # Set this flag to True when you want to use the test paths

if test:
    train_path = r"./cardDatabaseTest"
    eval_path = r"./evalTest"
else:
    train_path = r"./cardDatabase"
    eval_path = r"./evalSimilar"

os.makedirs(eval_path, exist_ok=True)

# Define the number of augmented images per card
num_augmented_images = 3

create_eval_data(train_path, eval_path, num_augmented_images)
