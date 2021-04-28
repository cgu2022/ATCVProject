import cv2
import numpy as np
import time

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
    
    return img

#--------------------------------------------------------

visited = None # Visited Array for DFS
dims = None
radius = 20 # Pixel Radius of Blob detector (I determined this experimentall)
size = 0 # variable to hold calculated blob size

def infest(i, j, img): # Mark the whole blob using DFS
    global visited, dims, size
    if visited[i][j] : # Already visited this pixel
        return
    else:
        size += 1
        visited[i][j] = True
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                newI = i+di
                newJ = j+dj
                if (newI > 0 and newI < dims[0] and newJ > 0 and newJ < dims[1]) and (img[newI][newJ]>128): # Make sure pixel exists & next pixel is white
                    infest(i+di, j+dj, img)
        
def findBlobs(img, microBlobFilter=True): # Count Number of blobs (subsitute for contour detection) | MicroFilter helps with blob counting
    global visited, dims, size
    height, width = img.shape
    dims = (height, width)
    visited = height * [width*[False]]
    blobs = []
    for i in range(height):
        for j in range(width):
            if img[i][j] == 255 and not visited[i][j]: # Found a new blob to count
                size = 0 # Reset blob size counter
                infest(i, j, img)
                print("Blob size:", size)
                if microBlobFilter and size < 20: # If microfilter is on, make sure counted blob is not mini, insignificant, noisy so that it won't mess up counting (I determined threshold experimentally)
                    continue
                blobs.append((i,j))
    return blobs # return blobs found

#--------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
analyze = True

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img = cap.read()
    cv2.imshow("img", img)

    if analyze:
        # Convert to Grayscale --> Threshold (to make calculations easy) --> Invert color scheme (because coutours are caluclated based on white figures)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale
        height, width = gray.shape
        #thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1] # Basic Thresholding - (if pixel > 90: --> white [255], else: --> black[0])
        thresholdValue, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU) # Adaptive Thresholding - threshold value based on histogram analysis
        print("Threshold Value used:", thresholdValue)
        thresh = cv2.bitwise_not(thresh) # Inverting color scheme
        cv2.imshow("thresh", thresh)
        cv2.imwrite("thresh.png", thresh)


        # Area Filter (to eliminate noise)
        areaFiltered = areaFilter(thresh)
        cv2.imshow("areaFiltered", areaFiltered)
        cv2.imwrite("areaFiltered.png", areaFiltered)

        # Getting Contours (for determining ROI)
        contours, h = cv2.findContours(areaFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Largest Contour will be first (if there so happens to be multiple contours)
        largestContour = contours[0] 

        # Getting ROI
        rect = cv2.boundingRect(largestContour)
        x,y,w,h = rect
        cropped = thresh[y:y+h, x:x+w]
        cv2.imshow("ROI", cropped)
        cv2.imwrite("ROI.png", cropped)

        # Slicing
        roiy, roix = cropped.shape
        top = cropped[0:(roiy//3),0:roix]
        mid = cropped[int(roiy*0.33):int(roiy*0.67),0:roix]
        bot = cropped[int(2*roiy//3):roiy,0:roix]
        cv2.imshow("top", top)
        cv2.imshow("mid", mid)
        cv2.imshow("bot", bot)
        cv2.imwrite("top.png", top)
        cv2.imwrite("mid.png", mid)
        cv2.imwrite("bot.png", bot)

        # Getting Blob Count of Slices
        print("\nTop:")
        blobtop = findBlobs(top)
        print("\nMid:")
        blobmid = findBlobs(mid)
        print("\nBot:")
        blobbot = findBlobs(bot)
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

    '''if cv2.waitKey(1):
        if 0xFF == ord('q'): # if press q, quit the program
            break
        else: # else, just toggle analyze
            if analyze:
                print("Turned off letter recognition!")
            else:
                print("Turned on letter recognition!")
            analyze = not analyze'''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(700)

cap.release()
cv2.destroyAllWindows()