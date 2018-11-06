import cv2 as cv
import numpy as np
import math
import sys

fn = "input1.png"

src = cv.imread(fn)

# canny edge detection and binarise
dst = cv.Canny(src, 50, 200)  
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)


#  for dilation as told by the mam.
kernel = np.ones((5,5),np.uint8)
cdst = cv.erode(cdst,kernel,iterations = 1)


# for hough lines , using probabilistic hough line transform function
lines = cv.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
a,b,c = lines.shape


for i in range(a):
    cv.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255)   )


# finally write it.
cv.imwrite('output1.png', cdst)

