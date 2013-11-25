#!/usr/bin/env python

import cv2
import numpy as np
import copy

cv2.namedWindow("mainWindow", cv2.CV_WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)

skin_min = np.array([0, 0, 70], np.uint8)
skin_max = np.array([20, 255, 255], np.uint8)

def get_image():
  ret, image = camera.read(0)
  blurred = cv2.GaussianBlur(image, (5, 5), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  return cv2.inRange(hsv, skin_min, skin_max)

def get_contours(image):
  contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  return contours

def get_threshold():
  thresh = get_image()
  return thresh

def get_figure():
  thresh = get_image()
  contours = get_contours(copy.copy(thresh))
  chosen = (0, None)
  for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    if chosen[0] < w * h:
      chosen = (w*h, contour)
  if chosen[1] != None:
    x,y,w,h = cv2.boundingRect(chosen[1])
    return thresh[y:y+h, x:x+w]
  else:
    return thresh

def show(image):
  cv2.imshow('mainWindow', image)

def compare_to(image, figure):
  result = cv2.matchTemplate(image, figure, cv2.cv.CV_TM_CCOEFF_NORMED)
  return result.max()

def compare(image):
  return { 'rock': compare_to(image, rock), 'paper': compare_to(image, paper), 'scissors': compare_to(image, scissors) }

scan_time = 2000
cv2.cv.CV_TM_CCOEFF_NORMED
print("Rock Paper Scissors")

print("scan ROCK")
cv2.waitKey(scan_time)
rock = get_figure()
show(rock)

print("scan PAPER")
cv2.waitKey(scan_time)
paper = get_figure()
show(paper)

print("scan SCISSORS")
cv2.waitKey(scan_time)
scissors = get_figure()
show(scissors)

i = 0
while True:
  thresh = get_threshold()
  show(thresh)
  i = i + 1
  if i % 10 == 0:
    comparision = compare(thresh)
    winner = { v:k for k, v in comparision.items() }[max(comparision.values())]
    print('Round %d' % i)
    print('Rock: %f' % comparision['rock'])
    print('Paper: %f' % comparision['paper'])
    print('Scissors: %f' % comparision['scissors'])
    print('WINNER: %s' % winner)
    print('')
