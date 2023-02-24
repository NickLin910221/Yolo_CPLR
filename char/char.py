import os
import cv2

for filename in os.listdir("C:\\Users\\NICK\\Desktop\\Practice\\yolov5_plate\\runs\\detect\\exp13\\crops\\plate"):
    if filename != "char.py":
        print(filename)
        cap = cv2.VideoCapture("C:\\Users\\NICK\\Desktop\\Practice\\yolov5_plate\\runs\\detect\\exp13\\crops\\plate\\" + filename)
        ret, raw = cap.read()
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", img)
        cv2.imwrite(filename, img)
        cv2.waitKey(0)