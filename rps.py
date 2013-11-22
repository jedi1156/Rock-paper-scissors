#!/usr/bin/env python

import cv2

print("Rock Paper Scissors")

cv2.namedWindow("mainWindow", cv2.CV_WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)

def repeat():
  ret, frame = camera.read()
  cv2.imshow("mainWindow", frame)

while True:
  repeat()
