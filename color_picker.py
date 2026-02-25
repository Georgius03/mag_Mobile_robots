import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Trackbars')

cv2.createTrackbar('H', 'Trackbars', 0, 180, nothing)
cv2.createTrackbar('S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('mH', 'Trackbars', 0, 180, nothing)
cv2.createTrackbar('mS', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('mV', 'Trackbars', 0, 255, nothing)

# cv2.setTrackbarPos('S', 'Trackbars', 160)
# cv2.setTrackbarPos('V', 'Trackbars', 135)

cv2.setTrackbarPos('mH', 'Trackbars', 180)
cv2.setTrackbarPos('mS', 'Trackbars', 255)
cv2.setTrackbarPos('mV', 'Trackbars', 255)

# img = cv2.imread('computer_vision_mag\images\Dataset_MobRob_V1\WIN_20260212_17_57_40_Pro.jpg', 1)
# img = cv2.resize(img, (800, 600))

cap = cv2.VideoCapture(0)

rainbow = cv2.imread('images/rainbow.jpg', 1)
rainbow = cv2.resize(rainbow, (800, 600))

while True:
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rainbow_hsv = cv2.cvtColor(rainbow, cv2.COLOR_BGR2HSV)

    H = cv2.getTrackbarPos('H', 'Trackbars')
    S = cv2.getTrackbarPos('S', 'Trackbars')
    V = cv2.getTrackbarPos('V', 'Trackbars')
    mH = cv2.getTrackbarPos('mH', 'Trackbars')
    mS = cv2.getTrackbarPos('mS', 'Trackbars')
    mV = cv2.getTrackbarPos('mV', 'Trackbars')

    mask = cv2.inRange(hsv, (H, S, V), (mH, mS, mV))
    # mask = cv2.dilate(mask, kernel=(6, 6), iterations=2)

    rainbow_mask = cv2.inRange(rainbow_hsv, (H, S, V), (mH, mS, mV))
    cv2.imshow('Mask', mask)

    masked_image = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('Trackbars', masked_image)
    cv2.imshow('test', masked_image)

    rainbow_show = cv2.bitwise_and(rainbow, rainbow, mask=rainbow_mask)
    cv2.imshow('Rainbow', rainbow_show)

    if cv2.waitKey(1) == ord('q'):
        break