import cv2
import numpy as np

# Setting the width and height of the output image
###########################################
imgWidth = 480
imgHeight = 640
###########################################


webcam = cv2.VideoCapture(1)
webcam.set(3, 640)
webcam.set(4, 480)

# Function to preprocess the image. It returns the edge map of the image
def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThresh = cv2.erode(imgDialation, kernel, iterations=1)
    return imgThresh


# A function to get contours from the image, and returning the biggest contour.
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    biggest = np.array([])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


# A function to reorder points according to the scanned document, to display the correct warped image.
def reorderPoints(myPoints):
    myPoints = myPoints.reshape(4, 2)
    newPoints = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(axis=1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    difference = np.diff(myPoints, axis=1)
    newPoints[1] = myPoints[np.argmin(difference)]
    newPoints[2] = myPoints[np.argmax(difference)]
    return newPoints

# A function to warp the image, and display the bird's eye view for the scanned document.
def getWarp(img, biggest):
    biggest = reorderPoints(biggest)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgOutput = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
    return imgOutput


# Loop for getting webcam feedback, and displaying the final output.
while True:
    success, frame = webcam.read()
    frame = cv2.resize(frame, (imgWidth, imgHeight))
    imgContour = frame.copy()

    imgThresh = preprocessing(frame)
    biggest = getContours(imgThresh)

    if biggest.size != 0:
        imgWarped = getWarp(frame, biggest)
    else:
        imgWarped = frame

    cv2.imshow("Webcam", imgWarped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

