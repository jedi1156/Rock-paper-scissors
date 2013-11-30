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

  #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  #cannied = cv2.GaussianBlur(gray,(3,3),0)
  #cannied = cv2.Canny(cannied,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)

  #if not isScanning():
  #cv2.accumulateWeighted(frame, scan, 0.005)

  #return result & cannied
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

scan_time = 2000
cv2.cv.CV_TM_CCOEFF_NORMED
print("Rock Paper Scissors")

moment_thresholds = (0.1, 0.05)
area_threshold = 90
class Pattern:
  def __init__(self, name):
    self.name = name
    print("scan %s" % name)
    cv2.waitKey(scan_time)
    self.img, self.contour = get_figure()
    self.moments = moments(self.contour)
    print(self.moments)

  def show(self):
    show(self.img)

  def detect(self, image):
    return cv2.matchShapes(self.contour, get_max_contour(image), cv2.cv.CV_CONTOURS_MATCH_I3, 0)

print("scan BACKGROUND")
cv2.waitKey(scan_time)
scan = read_image()
scan = np.float32(scan)

"""
rock = Pattern("ROCK")
paper = Pattern("PAPER")
scissors = Pattern("SCISSORS")
patterns = (rock, paper, scissors)
for p in patterns:
  p.show()
"""

history = dict()

while True:
  iteration += 1
  it_modulo = iteration % modulo
  if it_modulo == 12 or it_modulo == 16 or it_modulo == 0:
    beep()
  thresh = get_threshold()
  colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
  contour = get_max_contour(thresh)
  """
  x,y,w,h = cv2.boundingRect(contour)
  searchable = thresh[y:y+h, x:x+w]
  loc = cv2.matchTemplate(searchable, np.ones((5, 5), dtype=np.uint8), cv2.cv.CV_TM_CCORR)
  cv2.rectangle(colored, (x, y), (x+w, y+h), (0, 0, 255), 5)
  if contour != None:
    c,r = cv2.minEnclosingCircle(contour)
    print(c, r)
    cv2.circle(colored, (int(c[0]), int(c[1])), int(r), (255, 0, 0), 5)
  if contour != None and len(contour) > 5:
    ellipse = cv2.fitEllipse(contour)
    newx = ellipse[0][0] + (ellipse[1][0] - ellipse[0][0]) / 3
    newy = ellipse[0][1] + (ellipse[1][1] - ellipse[0][1]) / 3
    ellipse = (ellipse[0], (newx, newy), ellipse[2])
    cv2.ellipse(colored, ellipse, (255, 0, 0), 5)
    #cv2.circle(colored, (int(ellipse[0][0]), int(ellipse[1][0])), 10, (0, 255, 0), 10)
    #cv2.circle(colored, (int(ellipse[0][1]), int(ellipse[1][1])), 10, (255, 0, 0), 10)
    print(ellipse)
  """
  if contour != None:
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    if defects != None:
      for defect in defects:
        idx = defect[0][2]
        point = contour[idx]
        print(point)
        cv2.circle(colored, tuple(point[0]), 10, (0, 255, 255), 3)
      #cv2.drawContours(colored, [contour], 0, (0, 255, 0), 5)
      #cv2.drawContours(colored, [hull], 0, (255, 0, 0), 5)
      #print(defects)
      #cv2.drawContours(colored, [defects], 0, (255, 0, 0), 5)
  m = cv2.moments(thresh)
  area = m['m00']
  if area > 0.1:
    x = m['m10'] / area
    y = m['m01'] / area
    cv2.circle(colored, (int(x), int(y)), 10, (255, 0, 0), 10)
    if len(contour) > 5:
      ellipse = cv2.fitEllipse(contour)
      cv2.circle(colored, (int(ellipse[0][0]), int(ellipse[0][1])), 10, (0, 255, 0), 10)

  show(colored)
  continue
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
