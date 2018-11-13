import cv2
import numpy as np
from math import *
from matplotlib import pyplot as plt
# def find_skeleton3(img,iter=1000000):
#     skeleton = np.zeros(img.shape,np.uint8)
#     eroded = np.zeros(img.shape,np.uint8)
#     temp = np.zeros(img.shape,np.uint8)

#     _,thresh = cv2.threshold(img,127,255,0)

#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

#     iters = 0
#     while(True):
#         cv2.erode(thresh, kernel, eroded)
#         cv2.dilate(eroded, kernel, temp)
#         cv2.subtract(thresh, temp, temp)
#         cv2.bitwise_or(skeleton, temp, skeleton)
#         thresh, eroded = eroded, thresh # Swap instead of copy

#         iters += 1
#       # if cv2.countNonZero(thresh) == 0 or iters>=iter:
#         if iters>=iter:
#             return (skeleton,iters)

kernel = np.ones((2,2),np.uint8)
kernel9 = np.ones((3,3),np.uint8)
img = cv2.imread('input1.png')
#print((img));
#img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR);
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
high_th,img = cv2.threshold(img,230,255,cv2.THRESH_BINARY);
low_th = 0.5*high_th;
#plt.imshow(img);
#plt.show();
dilation = cv2.dilate(img,kernel,iterations=2)
#erosion = cv2.erode(dilation,kernel,iterations=10);

#edges = cv2.Canny(img,mu-0*sigma,mu+0*sigma)
edges = cv2.Canny(img,low_th,high_th);
plt.subplot(121),plt.imshow(edges);
#plt.show();
#lines = cv2.HoughLinesP(image=edges,rho=0.02,theta=np.pi/500,threshold=0);
dilation = cv2.dilate(edges,kernel,iterations=1)
#dilation = cv2.cvtColor(dilation,cv2.COLOR_BGR2GRAY);
high_th,dilation = cv2.threshold(dilation,230,255,cv2.THRESH_BINARY);
cdst = cv2.cvtColor(dilation,cv2.COLOR_GRAY2BGR);
dst = dilation
plt.subplot(122),plt.imshow(dilation);
#plt.show();

lines = cv2.HoughLinesP(dst, 1, pi/180.0, 10, np.array([]), 2, 0)
a,b,c = lines.shape
print(lines.size);
cdst = np.zeros_like(cdst);
# actually it gives two points (x1 ,y1) and (x2,y2) so that A line can be drawn with this. Actually we need
# points only ....so as to find two parallel lines and then finding mid line from this and then finding intersection.

for i in range(a):
    cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255)   )

plt.subplot(111),plt.imshow(cdst);

cv2.imwrite('output1.png', cdst)

# dst = cv2.cornerHarris(img,4,5,0.04);
# dst = cv2.dilate(dst,None);
# hh,th_corner = cv2.threshold(dst,1,255,cv2.THRESH_BINARY);
# plt.subplot(121),plt.imshow(dst,cmap='gray');
# plt.subplot(122),plt.imshow(th_corner,cmap='gray');
# plt.show();

#erosion = cv2.erode(dilation,kernel,iterations=100);
#opening = cv2.morphologyEx(dilation,cv2.MORPH_OPEN,kernel);
#closing = cv2.morphologyEx(erosion,cv2.MORPH_CLOSE,kernel9);
#(skel,it) = find_skeleton3(dilation,1000);
#dilation = cv2.dilate(skel,kernel,iterations=1)
#closing = cv2.morphologyEx(dilation,cv2.MORPH_OPEN,kernel9);
#edges = cv2.Canny(closing,mu-1*sigma,mu+1*sigma)
#th,dst = cv2.threshold(closing, 0,255, cv2.THRESH_BINARY);
#plt.imshow(closing);
#plt.show();
#a,b,c = lines.shape
#for i in range(a):
#    cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

#plt.subplot(221),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(222),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(223),plt.imshow(dilation,cmap = 'gray')
#plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(224),plt.imshow(closing,cmap = 'gray')
#plt.title('Closing Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(325),plt.imshow(opening,cmap = 'gray')
#plt.title('opening Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(326),plt.imshow(dst,cmap = 'gray')
#plt.title('threshold Image'), plt.xticks([]), plt.yticks([])

#imgplot = plt.imshow(erosion);
#plt.ion()
#plt.draw()
#plt.pause(0.001)

plt.show()
#plt.pause(0.001)
