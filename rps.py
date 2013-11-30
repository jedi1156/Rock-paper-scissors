#!/usr/bin/env python

import cv2
import numpy as np
import copy

cv2.namedWindow("mainWindow", cv2.CV_WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)

skin_min = np.array([0, 0, 70], np.uint8)
skin_max = np.array([20, 255, 255], np.uint8)
iteration = 0
modulo = 20

def read_image():
  ret, image = camera.read(0)
  return image

def get_image():
  global scan
  global iteration
  frame = read_image()

  result = frame.astype(np.int32)
  result = np.subtract(scan, result)
  result = np.abs(result)
  result = result.astype(np.uint8)
  result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
  result = cv2.inRange(result, np.array([0, 0, 50]), np.array([255, 255, 255]))

  if not isScanning():
    cv2.accumulateWeighted(frame, scan, 0.001)

  return result

def get_contours(image):
  contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  return contours

def get_max_contour(image):
  contours = get_contours(copy.copy(image))
  chosen = (0, None)
  for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    if chosen[0] < w * h:
      chosen = (w*h, contour)
  return chosen[1]

def get_threshold():
  thresh = get_image()
  return thresh

def get_figure():
  thresh = get_image()
  chosen = get_max_contour(thresh)
  if chosen != None:
    x,y,w,h = cv2.boundingRect(chosen)
    margin = 50
    w = w + 2 * margin
    h = h + 2 * margin
    x = x - margin
    y = y - margin
    if x < 0:
      x = 0
    if y < 0:
      y = 0
    return thresh[y:y+h, x:x+w]
  else:
    return thresh

def show(image):
  cv2.imshow('mainWindow', image)

def moments(image):
  moments = cv2.moments(get_max_contour(image))
  return cv2.HuMoments(moments)[0]

def compare_to(image, figure):
  img_moments = moments(image)
  fig_moments = moments(figure)
  return float(10 - abs(img_moments[0] - fig_moments[0]))

def compare(image):
  return { 'rock': compare_to(image, rock), 'paper': compare_to(image, paper), 'scissors': compare_to(image, scissors) }

def beep():
  print("\a") 

def isScanning():
 return iteration > 0 and iteration % modulo < 3

scan_time = 2000
cv2.cv.CV_TM_CCOEFF_NORMED
print("Rock Paper Scissors")

print("scan BACKGROUND")
cv2.waitKey(scan_time)
scan = read_image()
scan = np.float32(scan)

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

history = dict()

while True:
  iteration += 1
  it_modulo = iteration % modulo
  if it_modulo == 12 or it_modulo == 16 or it_modulo == 0:
    beep()
  thresh = get_threshold()
  show(thresh)
  if isScanning():
    comparision = compare(thresh)
    winner = { v:k for k, v in comparision.items() }[max(comparision.values())]
    print('Round %d' % iteration)
    print('Rock: %f' % comparision['rock'])
    print('Paper: %f' % comparision['paper'])
    print('Scissors: %f' % comparision['scissors'])
    print('WINNER: %s' % winner)
    print('')
    if not winner in history:
      history[winner] = 0
    history[winner] += 1
  elif it_modulo == 3:
    print(history)
  elif it_modulo == 4:
    history = dict()
