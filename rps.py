#!/usr/bin/env python

import cv2
import numpy as np

cv2.namedWindow("mainWindow", cv2.CV_WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)

skin_min = np.array([0, 0, 70], np.uint8)
skin_max = np.array([20, 255, 255], np.uint8)

def get_threshold():
  ret, image = camera.read()
  blurred = cv2.GaussianBlur(image, (5, 5), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  thresh = cv2.inRange(hsv, skin_min, skin_max)
  thresh_dup = cv2.inRange(hsv, skin_min, skin_max)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  thresh_dup = cv2.cvtColor(thresh_dup, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(thresh_dup, contours, -1, (0, 255, 0), 3)
  return thresh_dup

def show(image):
  cv2.imshow('mainWindow', image)

def compare_to(image, figure):
  return float(cv2.matchTemplate(image, figure, cv2.cv.CV_TM_CCOEFF_NORMED))

def compare(image):
  return { 'rock': compare_to(image, rock), 'paper': compare_to(image, paper), 'scissors': compare_to(image, scissors) }

scan_time = 2000
cv2.cv.CV_TM_CCOEFF_NORMED
print("Rock Paper Scissors")

print("scan ROCK")
cv2.waitKey(scan_time)
rock = get_threshold()
show(rock)

print("scan PAPER")
cv2.waitKey(scan_time)
paper = get_threshold()
show(paper)

print("scan SCISSORS")
cv2.waitKey(scan_time)
scissors = get_threshold()
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
