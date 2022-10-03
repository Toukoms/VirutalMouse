import cv2 as cv
import pyautogui
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector

#####################################
wCam, hCam = 640, 480
wScreen, hScreen = pyautogui.size()
#####################################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

handtrack = HandDetector(maxHands=1, detectionCon=0.7)

def find_deplacement(old_position, cur_position):
    print(abs(old_position-cur_position))

initialTime = time.time()

while True:
    timer = int(time.time()-initialTime)
    success, img = cap.read()
    
    img = cv.flip(img, 1)
    
    quart_w = int(wCam*1/4)
    tier_h = int(hCam*1/4)
    
    pts1 = (quart_w, tier_h)
    pts2 = (wCam-quart_w, hCam-tier_h)
    
    cv.rectangle(img, pts1, pts2, (150,10,10), 3)
    
    # find hand
    hand, img = handtrack.findHands(img, flipType=False)  # -> [[hand],[hand],...], img
    if hand:
        initialTime = time.time()
        hand = hand[0]
        
        lm_lists = hand['lmList']  # [[x1,y1,z1],[x2,y2,z2],...]
        x1, y1 = lm_lists[8][:-1]  # coordonné (x, y) index
        x2, y2 = lm_lists[12][:-1]  # coordonné (x, y) majeur
        
        fingers_up = handtrack.fingersUp(hand)  # -> [0, 0, 0, 1, 0] : ( 1 Up, 0 Down)
        
        # convertir le coordinnée en taille de la fenêtre
        x = np.interp(x1, (0, wCam), (0, wScreen))
        y = np.interp(y1, (0, hCam), (0, hScreen))
        
        # position actuelle de souris
        curX, curY = pyautogui.position()
            
        if fingers_up[1] == 1 and fingers_up[2] == 0 and pts1 < (x1, y1) < pts2:
            pyautogui.moveTo(x, y)
        
        distance, info = handtrack.findDistance((x1,y1), (x2,y2))

        if 0 < distance < 30:  # si index et majeur son proche
            pyautogui.mouseDown(curX, curY, 'left')
            pyautogui.mouseUp(curX, curY, 'left')
           
    cv.imshow('Image Original', img)
    if cv.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
