# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import pandas as pd

from .line_removal import remove_lines


def join_contours(thresh_gray, contours):
  #Join near morphology
  thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)));
  
  contours = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  sorted_cnts = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
  
  new_cnts = []

  count = 0
  for i in range(len(sorted_cnts)):
    new_cnts.append(sorted_cnts[i])
    if count >= 3:
      break
    count += 1

  for idx, c in enumerate(sorted_cnts):
    if idx >= 4:
      cv2.fillPoly(thresh_gray, pts=[c], color=0) 

  thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)));

  return thresh_gray

def get_digits_from_image(image):
  image = remove_lines(image)
  # Crop a bit
  image = image[int(0.05*image.shape[0]):int(0.95*image.shape[0]), int(0.05*image.shape[1]):int(0.95*image.shape[1])]

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # perform edge detection, find contours in the edge map, and sort the
  # resulting contours from left-to-right
  edged = cv2.Canny(blurred, 30, 150)
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  try:
    old_cnts = []
    if len(cnts) == 0:
      return None
    cnts = sort_contours(cnts, method="left-to-right")[0]
    old_cnts = cnts[:]
    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing

    #Join close enough contours
    debug_edged = edged.copy()
    if len(cnts) > 1:
      joined = join_contours(edged, cnts)
      cnts = cv2.findContours(joined, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      if len(cnts) == 0:
        return None
      cnts = sort_contours(cnts, method="left-to-right")[0]
  except Exception as e:
    return None

  chars = []

  for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    # filter out bounding boxes, ensuring they are neither too small
    # nor too large
    if (w >= 8 and w <= 250) and (h >= 17 and h <= 220):
      # extract the character and threshold it to make the character
      # appear as *white* (foreground) on a *black* background, then
      # grab the width and height of the thresholded image
      roi = gray[y:y + h, x:x + w]
      thresh = cv2.threshold(roi, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      (tH, tW) = thresh.shape
      # if the width is greater than the height, resize along the
      # width dimension
      if tW > tH:
        thresh = imutils.resize(thresh, width=32)
      # otherwise, resize along the height
      else:
        thresh = imutils.resize(thresh, height=32)

      # re-grab the image dimensions (now that its been resized)
      # and then determine how much we need to pad the width and
      # height such that our image will be 32x32
      (tH, tW) = thresh.shape
      dX = int(max(0, 32 - tW) / 2.0)
      dY = int(max(0, 32 - tH) / 2.0)
      # pad the image and force 32x32 dimensions
      padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0))
    
      padded = cv2.resize(padded, (32, 32))
      # prepare the padded image for classification via our
      # handwriting OCR model
      padded = padded.astype("float32") / 255.0
      padded = np.expand_dims(padded, axis=-1)

      # update our list of characters that will be OCR'd
      chars.append((padded, (x, y, w, h)))

  if len(chars) > 4:
    return None
  return chars