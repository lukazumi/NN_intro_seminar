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
import copy


SHAPE = (32, 32)
DEBUG = False
CROP = True


def classify_img(model_path):
    face_cascade = cv2.CascadeClassifier("har-cascaades/haarcascade_frontalface_default.xml")

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

    # build model
    f = open(os.path.join(model_path, 'model_architecture.json'), 'r')
    model = model_from_json(f.read())
    f.close()
    model.load_weights(os.path.join(model_path, 'model_weights.h5'))
    adam = Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # open cap webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img2 = cap.read()
        img2g = cv2.cvtColor(copy.copy(img2), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img2g, 1.3, 5)
        if len(faces) > 0:
            for idx, face in enumerate(faces):
                X = []
                (x, y, w, h) = face
                roi = img2g[
                      max(y - 25, 0):
                      min(y + h + 25, img2g.shape[0]),
                      max(x - 25, 0):
                      min(x + w + 25, img2g.shape[1])]
                np_final = misc.imresize(np.asarray(roi), size=SHAPE, interp='bilinear', mode=None)

                cv2.imwrite("fuck.jpg", np_final)

                X.append(np_final)
                X = np.float32(np.array(X) / 255.0)
                X = np.reshape(X, (len(X), 1, SHAPE[0], SHAPE[1]))
                # get top prediction
                out = ""
                predicted_classes = model.predict_classes(X, batch_size=1, verbose=0)
                for i, predicted_class in enumerate(predicted_classes):
                    out += classnames_lookup[str(predicted_class)]
                    cv2.putText(img2, out, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))

        cv2.imshow('frame', img2)
        # exit the camera loop prematurely with the q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', type=str,
                        help='The path of the directory containing the .json, .h5 and lookup.txt',
                        required=True)
    return parser.parse_args(argv)

if __name__ == "__main__":
    arg = parse_arguments(sys.argv[1:])
    classify_img(model_path=arg.model_dir)
