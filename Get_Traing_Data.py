"""
Luke Jackson 2017 I-Tap
"""

import urllib.request
import cv2
import numpy as np
import os
import argparse
import sys
import socket


def download_neg_images(linksTextFile):
    socket.setdefaulttimeout(10)  # times out non responding fetchs when downloading images
    pic_num = 1

    # setup directories
    img_dir = os.path.join(sys.path[0], "TrainingData")
    classname = linksTextFile[:linksTextFile.rfind(".txt")]
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(os.path.join(img_dir, classname)):
        os.makedirs(os.path.join(img_dir, classname))

    file = os.path.join(sys.path[0], linksTextFile)
    print(file)
    f = open(file, 'r', encoding='utf-8')
    imagenet_search_urls = f.read().splitlines()
    cnt = 0
    for imagenet_search_url in imagenet_search_urls:
        cnt = cnt+1
        try:
            print(str(cnt) + '/ ' + str(len(imagenet_search_urls)))
            # download image into temp
            urllib.request.urlretrieve(imagenet_search_url, os.path.join(img_dir, "tmp.jpg"))
            # read image into python
            img = cv2.imread(os.path.join(img_dir, "tmp.jpg"), cv2.IMREAD_GRAYSCALE)
            # really small images are simply discarded.
            if img.size > 20000:
                # make the images 100px x ~relative px
                # becasue pixel density has little affect on a models accuracy, little is a good optimizer.
                width, height = img.shape[1::-1]
                adj = 400 / width
                resized_image = cv2.resize(img, (400, int(round(height*adj))))
                # some sites return a "not found" placeholder image instead of timing out.
                # we need to avoid saving those.
                good = True
                for junk in os.listdir("junk"):
                    junk_img = cv2.imread(os.path.join("junk", junk))
                    if junk_img.shape == img.shape and not(np.bitwise_xor(junk_img, img).any()):
                        good = False
                # finally, if the image is good, we save it to the class folder.
                if good:
                    cv2.imwrite(os.path.join(img_dir, classname, str(pic_num) + ".jpg"), resized_image)
                    pic_num += 1
        except Exception as e:
            print(str(e))
    try:
        os.remove(os.path.join(img_dir, "tmp.jpg"))
    except OSError:
        pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('linksTextFile', type=str,
                        help='the text file containing all the links. The text files title will be used as classname (foldername)', required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    arg = parse_arguments(sys.argv[1:])
    download_neg_images(arg.linksTextFile)
