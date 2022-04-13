import keyboard
import cv2
import numpy as np
import math
import os
import time
import pyautogui
from point import p_crop
from hand import h_crop
from fin import f_crop
from fist import fs_crop
from thumbdown import t_crop
from okay import ok_crop

pyautogui.FAILSAFE = False
os.chdir( os.getcwd())
h1_cascade=cv2.CascadeClassifier('hand.xml')
okay_cascade = cv2.CascadeClassifier('ok.xml')
point_cascade = cv2.CascadeClassifier('point1.xml')
fin_cascade=cv2.CascadeClassifier('fin_2.xml')
fist_cascade=cv2.CascadeClassifier('fist.xml')
thumbdown_cascade = cv2.CascadeClassifier('thumbdown.xml')
cap = cv2.VideoCapture(0)
ca=0
count = 0
action = "None"
has_lost_once = 0
ok, frame = cap.read()
bbox = (50, 50, 150, 150)
fgbg1 = cv2.createBackgroundSubtractorKNN(200, 100, False)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):    
        _,img=cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        point=point_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        fin=fin_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        hand=h1_cascade.detectMultiScale(gray,1.1, 5)
        fist=fist_cascade.detectMultiScale(gray,1.3, 5)
        okay=okay_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,150))
        thumbdown=thumbdown_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        for (x,y,w,h) in okay:
                if action == "okay":
                        if count == 5:
                                pass
                        else:
                                count += 1
                else:
                        action = "okay"
                        count = 1

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                ok_crop(crop_img,img)
                print('okay')
                #time.sleep(0.5)
                
                
        for (x,y,w,h) in point:
                
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                p_crop(crop_img,img)
                if action != "point":
                        action = "point"
                        count = 1
                else:
                        if count == 5:
                                pyautogui.leftClick()
                                count+=1
                        elif count == 10:
                                pyautogui.doubleClick()
                        else:
                                count =count+1
                print('point ',action,  count)
                #time.sleep(0.5)

        for (x,y,w,h) in fin:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y-20:y+h+20,x-20:x+w+20]
                crop_img=img[y:y+h,x:x+w]
                h_crop(crop_img,img)

                if action == "fin":
                        if count == 5:
                                pyautogui.keyDown('altleft')
                                pyautogui.press('f4')
                                pyautogui.keyUp('altleft')
                                count+=1
                        else:
                                count += 1
                else:
                        action = "fin"
                        count = 1
                """if 320>x>0 and 240>y>0:
                        xa, ya = -0.25*(160-x),-0.25*(120-y)
                        pyautogui.move(xa, ya)
                        print(xa, ", ", ya)"""
                
                print('fin')

                #time.sleep(0.5)
        

        for (x,y,w,h) in thumbdown:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                t_crop(crop_img,img)
                if action == "thumb":
                        if count == 5:
                                pyautogui.keyDown('winleft')
                                pyautogui.press('d')
                                pyautogui.keyUp('winleft')
                        else:
                                count += 1
                else:
                        action = "thumb"
                        count = 1
                print('thumbdown')
                #time.sleep(0.5)

        for (x,y,w,h) in fist:
                track_window = (x,y,w,h)
                roi = img[y:y+h, x:x+w]
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                #fgmask1 = fgbg1.apply(hsv_roi);

                lower_skin = np.array([0,  58, 60])
                higher_skin = np.array([50, 174, 255])
                mask = cv2.inRange(hsv_roi, lower_skin, higher_skin)
                roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                pos = pyautogui.position()
                img = cv2.putText(img, str(pos[0]) + " and " + str(pos[1]), (10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                if has_lost_once == 1:
                        cv2.imwrite('moveinit.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                f_crop(crop_img,img)

                x_cent, y_cent = x+0.5*w, y+0.5*h
                if action == 'fist':
                        pyautogui.move(-4*(xs-x_cent), -4*(ys-y_cent))
                        xs, ys = x_cent, y_cent
                else:
                        action = 'fist'
                        count = 1
                        xs, ys = x_cent, y_cent
                print('fist')
                has_lost_once = 0

        if action == 'fist' and len(fin) == 0:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

                ret, bbox = cv2.meanShift(dst, track_window, term_crit)
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                masky = cv2.inRange(img[y:y+h, x:x+w], lower_skin, higher_skin)

                if not np.any(masky):
                        has_lost_once = 0
                        action = 'None'
                else:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                        x_cent, y_cent = x+0.5*w, y+0.5*h
                        if action == 'fist':
                                pyautogui.move(-4*(xs-x_cent), -4*(ys-y_cent))
                                xs, ys = x_cent, y_cent
                        else:
                                action = 'fist'
                                count = 1
                                xs, ys = x_cent, y_cent
                        if has_lost_once == 10:
                                pos = pyautogui.position()
                                img = cv2.putText(img, str(pos[0]) + " and " + str(pos[1]), (10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                                cv2.imwrite('moveend.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                has_lost_once = 0
                                action = 'None'
                                #tracker.clear()
                        else:
                                has_lost_once += 1

        
        cv2.imshow('Feed',img)
                
        k=cv2.waitKey(10)
        if k==27:
                break
cv2.destroyAllWindows()

    
