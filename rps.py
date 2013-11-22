#!/usr/bin/env python

import cv2
import numpy as np

print("Rock Paper Scissors")

cv2.namedWindow("mainWindow", cv2.CV_WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)

skin_min = np.array([0, 0, 70], np.uint8)
skin_max = np.array([20, 255, 255], np.uint8)

def repeat():
  ret, image = camera.read()

  blurred = cv2.GaussianBlur(image, (5, 5), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  thresh = cv2.inRange(hsv, skin_min, skin_max)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(image,contours, -1, (0, 255, 0), 3)

  cv2.imshow('mainWindow', image)

while True:
  repeat()
