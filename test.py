from pprint import pprint
import cv2 as cv
import mediapipe as mp
import numpy as np

MAX_NUM_HANDS = 2
MIN_DETECTION_CON = 0.5
MIN_TRACKING_CON = 0.5
FINGERS_NAME = ['thumb', 'index', 'middle', 'ring', 'pinky']

HANDS_CON = mp.solutions.hands_connections
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CON,
    min_tracking_confidence=MIN_TRACKING_CON,
)

mpDraw = mp.solutions.drawing_utils

def findHands(img, flipType=False):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    h, w, c = img.shape
    allHands = []
    
    if results.multi_hand_landmarks:
        for handType, handLms, hand3D in zip(results.multi_handedness, results.multi_hand_landmarks, results.multi_hand_world_landmarks):
            myHand = {}
            ## lmList
            mylmList = []
            lmFingers = {}
            lm3DList = []
            lm3DFingers = {}
            xList = []
            yList = []
            counter = 0
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([px, py, pz])
                xList.append(px)
                yList.append(py)
                # print('counter: ', counter, 'id: ', id, 'Mod id: ', id % 4)
                if (id % 4) == 3:
                    lmFingers[FINGERS_NAME[counter]] = mylmList
                    mylmList = []
                    counter += 1
            counter = 0
            for id, lm3D in enumerate(hand3D.landmark):
                px, py, pz = round(lm3D.x, 3), round(lm3D.y, 3), round(lm3D.z, 3)
                lm3DList.append([px, py, pz])
                if id % 4 == 3:
                    lm3DFingers[FINGERS_NAME[counter]] = lm3DList
                    lm3DList = []
                    counter += 1

            ## bbox
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), \
                        bbox[1] + (bbox[3] // 2)

            myHand["lmList"] = mylmList
            myHand["lm3DList"] = lm3DList
            myHand["lmFingers"] = lmFingers
            myHand["lm3DFingers"] = lm3DFingers
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)

            if flipType:
                if handType.classification[0].label == "Right":
                    myHand["type"] = "Left"
                else:
                    myHand["type"] = "Right"
            else:
                myHand["type"] = handType.classification[0].label
            allHands.append(myHand)
            
            # draw
            mpDraw.draw_landmarks(img, handLms, HANDS_CON.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec((20, 220, 220)),
                                  mpDraw.DrawingSpec((155, 121, 112)))
            cv.rectangle(img, (xmin-10, ymin-10), (xmax+10, ymax+10), (10, 64, 132), 2)
    
    return allHands

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    
    all_hand = findHands(img)
    if all_hand:
        hand1 = all_hand[0]

        print(f"Index Lm: {hand1['lmFingers']['index']}")
        print(f"Index Lm3D: {hand1['lm3DFingers']['index']}")
    
    cv.imshow('Original Image', img)
    
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
