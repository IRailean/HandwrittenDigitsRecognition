import cv2

def remove_horizontal_lines(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Remove horizontal
  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
  detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
  cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      cv2.drawContours(image, [c], -1, (255,255,255), 2)

  # Repair image
  repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
  result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

  return result

  
def remove_vertical_lines(image):
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Remove vertical
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
  detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
  cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
  for c in cnts:
      cv2.drawContours(image, [c], -1, (255,255,255), 2)

  # Repair image
  repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
  result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

  return result

def remove_lines(image):
  return remove_vertical_lines(remove_horizontal_lines(image))


