"""
Luke Jackson I-Tap 2017
"""


import cv2
import numpy as np

img = cv2.imread('img/cat1.jpg')

kernel_identity = np.array(
    [
        [0,0,0],
        [0,1,0],
        [0,0,0]
    ]
)

kernel_edge1 = np.array(
    [
        [1,0,-1],
        [0,0,0],
        [-1,0,1]
    ]
)

kernel_edge2 = np.array(
    [
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ]
)

kernel_edge3 = np.array(
    [
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
    ]
)

kernel_sharpen = np.array(
    [
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ]
)

img_identy = cv2.filter2D(img, -1, kernel_identity)
cv2.imwrite("cat1_identity.jpg", img_identy)

img_identy = cv2.filter2D(img, -1, kernel_edge1)
cv2.imwrite("cat1_edge1.jpg", img_identy)

img_identy = cv2.filter2D(img, -1, kernel_edge2)
cv2.imwrite("cat1_edge2.jpg", img_identy)

img_identy = cv2.filter2D(img, -1, kernel_edge3)
cv2.imwrite("cat1_edge3.jpg", img_identy)

img_identy = cv2.filter2D(img, -1, kernel_sharpen)
cv2.imwrite("cat1_sharp.jpg", img_identy)