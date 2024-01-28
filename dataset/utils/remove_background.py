import rembg
import numpy as np
from PIL import Image
import os


DATASET_PATH = 'archive/cars_train/cars_train/'
SAVE_PATH = 'removed_background_cars/cars_train/'

all_files = os.listdir(DATASET_PATH)

for idx, file in enumerate(all_files):
    # Load the input image
    input_image = Image.open(DATASET_PATH + file)

    # Convert the input image to a numpy array
    input_array = np.array(input_image)

    # Apply background removal using rembg
    output_array = rembg.remove(input_array)

    # Create a PIL Image from the output array
    output_image = Image.fromarray(output_array)
    new_image = Image.new("RGBA", output_image.size, "WHITE") # Create a white rgba background
    new_image.paste(output_image, (0, 0), output_image)

    # Save the output image
    new_image.convert('RGB').save(SAVE_PATH + file)
