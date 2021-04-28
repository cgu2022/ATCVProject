import cv2
import numpy as np
import time
from collections import deque

#--------------------------------------------------------

def areaFilter(img):
    contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Get all contours for areaFilter to calculate area via contours
    
    # Calculating areas of each contour
    areas = []
    for i in range(0,len(contours)):        
        area = cv2.contourArea(contours[i])
        areas.append(area)

    print("Detected Contour Areas:", areas)

    if len(areas) > 0:
        correctContour = areas.index(max(areas)) # Index of correct contour (the contour that contains the letter)
        for i in range(0, len(contours)):
            if i != correctContour:
                cv2.drawContours(img, contours, i, (0,0,0), cv2.FILLED) # Purging noise by filling in tiny contours
    
    return 
    
visited = None # Visited Array for DFS
dims = None
radius = 20 # Pixel Radius of Blob detector (I determined this experimentall)
purgeRadius = 20
size = 0 # variable to hold calculated blob size

def purge(i, j, img): # Removing blob using BFS since too many recursion calls for DFS
    global visited
    q = deque()
    q.append((i,j))
    while len(q) > 0:
        current = q[0]
        #print("Current:", current)
        q.popleft()  
        if not visited[current[0]][current[1]]:
            visited[current[0]][current[1]] = True
            img[current[0], current[1]] = 0 # Fill it black  
            for di in range(-purgeRadius, purgeRadius+1):
                    for dj in range(-purgeRadius, purgeRadius+1):
                        newI = current[0]+di
                        newJ = current[1]+dj
                        if (newI > 0 and newI < dims[0] and newJ > 0 and newJ < dims[1]) and (img[newI][newJ]>128): # Make sure pixel exists & next pixel is white
                            q.append((newI, newJ))

# Will take some time
def cleanBlobs(img, blobs):
    global dims, visited
    height, width = img.shape
    dims = (height, width)
    visited = height * [width*[False]]
    for location in blobs:
        purge(location[0], location[1], img)
        cv2.imshow("img", img)
        cv2.waitKey()
    return img

#--------------------------------------------------------

def infest(i, j, img): # Counts the whole blob's area using DFS (but does not remove the blob)
    global visited, dims, size
    if visited[i][j] : # Already visited this pixel
        return
    else: # Have not visited this pixel yet
        size += 1
        visited[i][j] = True
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                newI = i+di
                newJ = j+dj
                if (newI > 0 and newI < dims[0] and newJ > 0 and newJ < dims[1]) and (img[newI][newJ]>128): # Make sure pixel exists & next pixel is white
                    infest(newI, newJ, img)
        
def findBlobs(img, microBlobFilter=True): # Count Number of blobs (subsitute for contour detection) | MicroFilter helps with blob counting
    global visited, dims, size
    height, width = img.shape
    dims = (height, width)
    visited = height * [width*[False]]
    blobs = []
    areas = []
    for i in range(height):
        for j in range(width):
            if img[i][j] == 255 and not visited[i][j]: # Found a new blob to count
                size = 0 # Reset blob size counter
                infest(i, j, img)
                print("Blob size:", size)
                if microBlobFilter and size < 20: # If microfilter is on, make sure counted blob is not mini, insignificant, noisy so that it won't mess up counting (I determined threshold experimentally)
                    continue
                blobs.append((i,j))
                areas.append(size)
    return blobs, areas # return blobs found

#--------------------------------------------------------

# Reading img in
img = cv2.imread("R-inked-smallest.jpg")
#img = cv2.imread("B-smallest.jpg")
#img = cv2.imread("G-smallest.jpg")
cv2.imshow("img", img)

# Convert to Grayscale --> Threshold (to make calculations easy) --> Invert color scheme (because coutours are caluclated based on white figures)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale
height, width = gray.shape
#thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1] # Basic Thresholding - (if pixel > 90: --> white [255], else: --> black[0])
thresholdValue, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU) # Adaptive Thresholding - threshold value based on histogram analysis
print("Threshold Value used:", thresholdValue)
thresh = cv2.bitwise_not(thresh) # Inverting color scheme
cv2.imshow("thresh", thresh)


# Area Filter (to eliminate noise)
'''
areaFiltered = areaFilter(thresh)
cv2.imshow("areaFiltered", areaFiltered)
'''
blobs, blobAreas = findBlobs(thresh, microBlobFilter=False) # Don't want to filter because we want to find the noisy blobs
print("Blob Areas:", blobAreas)
print("Blob Locations:", blobs)
largestBlobIndex = blobAreas.index(max(blobAreas)) # Keep the largest blob
blobs.pop(largestBlobIndex)
areaFiltered = cleanBlobs(thresh, blobs) # Clean noisy blobs
cv2.imshow("areaFiltered", areaFiltered)
cv2.imwrite("areaFilteredDebug.png", areaFiltered)


# Getting Contours (for determining ROI)
contours, h = cv2.findContours(areaFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Largest Contour will be first (if there so happens to be multiple contours)
largestContour = contours[0] 

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

# Getting Blob Count of Slices
print("\nTop:")
blobtop, dummy = findBlobs(top)
print("\nMid:")
blobmid, dummy = findBlobs(mid)
print("\nBot:")
blobbot, dummy = findBlobs(bot)
print("\nTop: {}, Mid: {}, Bot: {}".format(len(blobtop), len(blobmid), len(blobbot)))

# Letter Detection using Blob Counts
if len(blobtop) == 1 and len(blobmid) == 1 and len(blobbot) == 2:
    print("Letter detected: R")
elif len(blobtop) == 1 and len(blobmid) == 1 and len(blobbot) == 1:
    print("Letter detected: B")
elif len(blobtop) == 1 and len(blobmid) == 2 and len(blobbot) == 1:
    print("Letter detected: G")
else:
    print("Letter detected: None")

cv2.waitKey()
cv2.destroyAllWindows()