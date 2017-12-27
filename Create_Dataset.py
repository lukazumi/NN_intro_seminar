"""
Luke Jackson I-Tap 2017
"""

import os
import sys
from scipy import misc
import numpy as np
import argparse
import shutil
from shutil import copyfile


SHAPE = (32, 32)  # images will re-scaled to this size.


def create_dataset(data_set_path, output_dataset_name):
    # folder names are class names
    folder_names = []
    for name in sorted(os.listdir(data_set_path)):
        if os.path.isdir(os.path.join(data_set_path, name)):
            folder_names.append(name)

    # files names are random(u-id), so sorting mixes it up a bit.
    folder_files = []
    for folder_name in folder_names:
        folder_files.append(sorted(os.listdir(os.path.join(data_set_path, folder_name))))

    number_of_classes = len(folder_names)
    # All classes have the same number of elements (data from data_gen py)
    number_of_examples_per_class = len(folder_files[0])

    data_imgs = []
    data_lbls = []

    # Load the images and labels into numpy arrays
    for i in range(number_of_classes):
        print(str(i) + "/" + str(number_of_classes))
        for j in range(number_of_examples_per_class):
            try:
                image_location = os.path.join(
                    data_set_path, folder_names[i], folder_files[i][j])
            except IndexError:
                print("this folder doesnt have enough images")
                print(folder_names[i])
                return
            image = misc.imread(image_location)
            image = misc.imresize(image, size=SHAPE, interp='bilinear', mode=None)
            #misc.imsave("resize.jpg", image)
            data_imgs.append(image)
            data_lbls.append(folder_names[i])

    # Turn the samples into proper numpy array of type
    # float32 (for use with GPU) rescaled in [0,1] interval.
    data_imgs = np.float32(np.array(data_imgs)/255.0)
    data_lbls = np.int32(np.array(data_lbls))
    number_of_classes = len(folder_names)

    # Make so that each batch of size "number_of_classes" samples is
    # balanced with respect to classes.
    # That is, each batch of size "number_of_classes" samples
    # will contain exactly one sample of each class.
    # In this way, when we split the data into train, validation, and test
    # datasets, all of them will be balanced with respect to classes
    # as long as the sizes of all of them are divisible by "number_of_classes".
    data_imgs = np.concatenate(
        [data_imgs[i::number_of_examples_per_class]
            for i in range(number_of_examples_per_class)])
    data_lbls = np.concatenate(
        [data_lbls[i::number_of_examples_per_class]
            for i in range(number_of_examples_per_class)])

    dataset_size = number_of_classes * number_of_examples_per_class

    # train - validation - test split is 80% - 10% - 10%
    # We also assume that the dataset_size is divisible by 10.
    data_imgs_train = data_imgs[:(dataset_size*8)//10]
    data_lbls_train = data_lbls[:(dataset_size*8)//10]

    data_imgs_val = data_imgs[(dataset_size*8)//10:(dataset_size*9)//10]
    data_lbls_val = data_lbls[(dataset_size*8)//10:(dataset_size*9)//10]

    data_imgs_test = data_imgs[(dataset_size*9)//10:]
    data_lbls_test = data_lbls[(dataset_size*9)//10:]

    f = open(os.path.join(data_set_path, output_dataset_name), "wb")
    np.save(f, data_imgs_train)
    np.save(f, data_lbls_train)
    np.save(f, data_imgs_val)
    np.save(f, data_lbls_val)
    np.save(f, data_imgs_test)
    np.save(f, data_lbls_test)
    np.save(f, number_of_classes)
    f.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir', type=str, help='The path that you have your image folders in', required=True)
    parser.add_argument('-output_name', type=str, help='The path and filename to output the data package to',
                        required=True)
    return parser.parse_args(argv)

if __name__ == "__main__":
    arg = parse_arguments(sys.argv[1:])
    create_dataset(arg.images_dir, arg.output_name)
