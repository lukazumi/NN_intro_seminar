"""
Luke Jackson I-Tap 2017
"""

import sys
import numpy as np
import os
import argparse
from scipy import misc
from keras.models import model_from_json
from keras.optimizers import Adam
import csv
import cv2


SHAPE = (32, 32)
DEBUG = False
CROP = True


def classify_img(model_path, images_dir=None):
    face_cascade = cv2.CascadeClassifier("haar-cascades/haarcascade_frontalface_default.xml")
    # graph was trained on integer class names instead of kanji. convert them back with this dictionary.
    classnames_lookup = {}
    dip = os.path.join(model_path, 'class_lookup.txt')
    with open(dip, encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                int(row[0])
            except ValueError:
                break
            classnames_lookup[row[0].rstrip()] = row[1].rstrip()

    f = open(os.path.join(model_path, 'model_architecture.json'), 'r')
    model = model_from_json(f.read())
    f.close()

    model.load_weights(os.path.join(model_path, 'model_weights.h5'))
    image_list = []
    for file in [f for f in os.listdir(images_dir) if f.endswith('.jpg')]:
        image_list.append(os.path.join(images_dir, file))

    X = []
    if CROP:
        for image_location in image_list:
            image = misc.imread(image_location, flatten=True)
            image = misc.imresize(image, size=image.shape, interp='bilinear', mode=None)
            faces = face_cascade.detectMultiScale(image, 1.3, 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                roi = image[
                      max(y-25, 0):
                      min(y + h+25, image.shape[0]),
                      max(x-25, 0):
                      min(x + w+25, image.shape[1])]
                np_final = misc.imresize(np.asarray(roi), size=SHAPE, interp='bilinear', mode=None)
                X.append(np_final)
            else:  # the case that the har-cascaade doesnt find a match
                print("face not found:" + image_location)
                image = misc.imread(image_location, flatten=True)
                image = misc.imresize(image, size=SHAPE, interp='bilinear', mode=None)
                X.append(image)
    else:
        for image_location in image_list:
            image = misc.imread(image_location, flatten=True)
            image = misc.imresize(image, size=SHAPE, interp='bilinear', mode=None)
            X.append(image)

    X = np.float32(np.array(X)/255.0)
    X = np.reshape(X, (len(X), 1, SHAPE[0], SHAPE[1]))

    adam = Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    print("-- top predictions--")
    predicted_classes = model.predict_classes(X, batch_size=1, verbose=1)
    print()
    for i, predicted_class in enumerate(predicted_classes):
        print(image_list[i], classnames_lookup[str(predicted_class)])
    print()
    print("-- detailed results --")

    prediction_probabilities = model.predict_proba(X, batch_size=1, verbose=1)
    for i, probs in enumerate(prediction_probabilities):
        ind = np.argsort(probs)
        out_string = classnames_lookup[str(ind[-1])].split(",")[0] + ": " + str(probs[ind[-1]]*100) + "%, " + classnames_lookup[str(ind[-2])].split(",")[0] + ": " + str(probs[ind[-2]]*100) + "%"
        # out = os.path.basename(image_list[i]) + out_string
        out = str(image_list[i]) + "> " +out_string
        print()
        print(out)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', type=str,
                        help='The path of the directory containing the .json, .h5 and lookup.txt',
                        required=True)
    parser.add_argument('-images_dir', type=str,
                        help='a directory containing images to classify. Each file in the directory is classified.',
                        required=True)
    return parser.parse_args(argv)

if __name__ == "__main__":
    arg = parse_arguments(sys.argv[1:])
    classify_img(model_path=arg.model_dir, images_dir=arg.images_dir)
