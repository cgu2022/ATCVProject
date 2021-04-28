import cv2
import numpy as np
import time

def areaFilter(img):
    contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Get all contours for areaFilter to calculate area via contours
    
    # Calculating areas of each contour
    areas = []
    for i in range(0,len(contours)):        
        area = cv2.contourArea(contours[i])
        areas.append(area)

    print("Areas:", areas)

    if len(areas) > 0:
        correctContour = areas.index(max(areas)) # Index of correct contour (the contour that contains the letter)
        for i in range(0, len(contours)):
            if i != correctContour:
                cv2.drawContours(img, contours, i, (0,0,0), cv2.FILLED)
    
    return img

#--------------------------------------------------------

# Reading img in
#img = cv2.imread("R-inked-small.jpg")
#img = cv2.imread("B-small.jpg")
img = cv2.imread("G-small.jpg")
cv2.imshow("img", img)

# Convert to Grayscale --> Threshold (to make calculations easy) --> Invert color scheme (because coutours are caluclated based on white lines)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.bitwise_not(thresh)
cv2.imshow("thresh", thresh)


# Area Filter (to eliminate noise)
areaFiltered = areaFilter(thresh)
cv2.imshow("areaFiltered", areaFiltered)

# Getting Contours (for ROI)
contours, h = cv2.findContours(areaFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Largest Contour will be first
largestContour = contours[0] 
print("Contours:", len(contours))
'''
blank =  np.zeros((height, width, 3))
blank = cv2.drawContours(blank.copy(), contours, 0, (0,255,0), 3)
cv2.imshow("blank", blank)
'''

# Getting ROI
rect = cv2.boundingRect(largestContour)
x,y,w,h = rect
cropped = thresh[y:y+h, x:x+w]
cv2.imshow("ROI", cropped)

# Slicing
roiy, roix = cropped.shape
top = cropped[0:(roiy//3),0:roix]
mid = cropped[int(roiy*0.33):int(roiy*0.67),0:roix]
bot = cropped[int(2*roiy//3):roiy,0:roix]

cv2.imshow("top", top)
cv2.imshow("mid", mid)
cv2.imshow("bot", bot)

# Getting Contours of Slices
(contop, h) = cv2.findContours(top.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(conmid, h) = cv2.findContours(mid.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(conbot, h) = cv2.findContours(bot.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

'''
blank =  np.zeros((int(roiy*0.67)-int(roiy*0.33), roix, 3))
blank = cv2.drawContours(blank.copy(), conmid, -1, (0,255,0), 3)
cv2.imshow("blank", blank)
'''

print("Top: {}, Mid: {}, Bot: {}".format(len(contop), len(conmid), len(conbot)))

if len(contop) == 1 and len(conmid) == 1 and len(conbot) == 2:
    print("Letter detected: R")
elif len(contop) == 1 and len(conmid) == 1 and len(conbot) == 1:
    print("Letter detected: B")
elif len(contop) == 1 and len(conmid) == 2 and len(conbot) == 1:
    print("Letter detected: G")
else:
    print("Letter detected: None")

cv2.waitKey()
#cap.release()
cv2.destroyAllWindows()