import os
import random
from scipy import ndarray
import numpy as np
import skimage as sk
from skimage import util
from skimage import io
from skimage.transform import resize
from scipy.ndimage.interpolation import rotate


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def random_rotation(image, angle_range=(0, 90)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image


def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), mask_value='random'):
    image = np.copy(image_origin)
    if np.random.rand() > p:
        return image
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image


# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'random_erasing': random_erasing
}

folder_path = input("Folder path (relative from current directory: ")
num_files_desired = int(input("Number of desired files: "))

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)

    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None

    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

    new_file_path = '%s/augmented_image_%s_%s.png' % (folder_path, image_name, num_generated_files)

    # write image to the disk
    transformed_image = sk.img_as_ubyte(transformed_image)
    io.imsave(new_file_path, transformed_image)

    num_generated_files += 1
