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

lowThreshold = 30
max_lowThreshold = 100
ratio = 3
kernel_size = 3
def get_image():
  global scan
  global iteration
  frame = read_image()

  result = frame.astype(np.int32)
  result = np.subtract(scan, result)
  result = np.abs(result)
  result = result.astype(np.uint8)
  result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
  result = cv2.inRange(result, np.array([0, 0, 10]), np.array([255, 255, 255]))
  result = cv2.erode(result, np.ones((10, 10)))
  result = cv2.dilate(result, np.ones((20, 20)))

  #if not isScanning():
  #cv2.accumulateWeighted(frame, scan, 0.005)

  return result


def get_contours(image):
  contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  return contours

def get_max_contour(image):
  contours = get_contours(copy.copy(image))
  chosen = (0, None)
  for contour in contours:
    if chosen[0] < len(contour):
      chosen = (len(contour), contour)
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
    return (thresh[y:y+h, x:x+w], chosen)
  else:
    return (thresh, None)

def show(image):
  cv2.imshow('mainWindow', image)

def moments(contour):
  ms = cv2.HuMoments(cv2.moments(contour))
  return (ms[0][0], ms[1][0])

def compare_to(image, figure):
  return figure.detect(image)

def compare(image):
  return { 'rock': compare_to(image, rock), 'paper': compare_to(image, paper), 'scissors': compare_to(image, scissors) }

def beep():
  print("\a") 

def isScanning():
 return iteration > 0 and iteration % modulo < 3

def detect_palm(contour):
  ellipse = cv2.fitEllipse(contour)
  center = (int(ellipse[0][0]), int(ellipse[0][1]))
  size   = (int(ellipse[1][0]), int(ellipse[1][1]))
  angle  = ellipse[2] - 90
  if angle < 0:
    angle += 180
  end1   = (int(center[0] - np.cos(np.radians(angle)) * size[1] / 2), int(center[1] - np.sin(np.radians(angle)) * size[1] / 2))
  mask_center = (center[0] + (end1[0] - center[0]) / 2, center[1] + (end1[1] - center[1]) / 2)
  mask_radius = int(size[1] * 0.375)
  return (mask_center, mask_radius)

def mask_palm(image, circle):
  mask = np.zeros(image.shape, dtype=np.uint8)
  cv2.circle(mask, circle[0], circle[1], 1, -1)
  return image * mask

print("Rock Paper Scissors")

class Pattern:
  def __init__(self, name):
    self.name = name
    print("scan %s" % name)
    cv2.waitKey(scan_time)
    self.img = get_image()
    palm = detect_palm(get_max_contour(self.img))
    self.img = mask_palm(self.img, palm)
    self.contour = get_max_contour(self.img)
    self.moments = moments(self.contour)
    print(self.moments)

  def show(self):
    show(self.img)

  def detect(self, image):
    return cv2.matchShapes(self.contour, get_max_contour(image), cv2.cv.CV_CONTOURS_MATCH_I3, 0)

scan_time = 2000
testing = False

print("scan BACKGROUND")
cv2.waitKey(scan_time)
scan = read_image()
scan = np.float32(scan)

if not testing:
  rock = Pattern("ROCK")
  rock.show()
  paper = Pattern("PAPER")
  paper.show()
  scissors = Pattern("SCISSORS")
  scissors.show()
  patterns = (rock, paper, scissors)

  history = dict()

while True:
  iteration += 1
  it_modulo = iteration % modulo
  if it_modulo == 12 or it_modulo == 16 or it_modulo == 0:
    beep()
  thresh = get_threshold()
  contour = get_max_contour(thresh)
  if contour != None and len(contour) > 5:
    center, radius = detect_palm(contour)
    masked = mask_palm(thresh, (center, radius))
    thresh = copy.copy(masked)

    colored = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(colored, [contour], 0, (255, 0, 0), 5)
    cv2.circle(colored, center, 10, (0, 0, 255), 5)
    cv2.circle(colored, center, radius, (0, 0, 255), 5)

    show(colored)
  if testing:
    continue
  if isScanning():
    comparision = compare(thresh)
    winner = { v:k for k, v in comparision.items() }[min(comparision.values())]
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
