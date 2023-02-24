import cv2
import time
import torch
import pytesseract
import queue

# rtmp = "rtmp://127.0.0.1:1935/live/mystream"
cap = cv2.VideoCapture(0)
fps = 0
q = queue.Queue()

# Web Camera
# cap = cv2.VideoCapture("C:\\Users\\NICK\\Documents\\license\\M8-5768.jpg")

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

#Model
model_plate = torch.hub.load('yolov5_plate1', 'custom', path='yolov5_plate1/runs/train/exp/weights/best.pt', source='local') # local repo # custom
model_char = torch.hub.load('yolov5_char', 'custom', path='yolov5_char/runs/train/exp/weights/best.pt', source='local') # local repo # custom

def decode(code):
    if code < 10:
        return str(code)
    elif code >= 10 and code < 35:
        return str(chr(code + 55))
    else:
        return ""

def plate_detect(img):    
    results = model_char(img)
    
    # Results
    coordinate = results.xywh[0]
    tmp = []
    for char in coordinate:
        tmp.append([int(char[0]), int(char[1]), int(char[2]), int(char[3]), float(char[4]), int(char[5])])
    tmp.sort(key=lambda row: (row[0]), reverse=False)
    # for i in range(len(tmp) - 1):
    #     if (tmp[i][0] > tmp[i + 1][0] and tmp[i][0] < tmp[i + 1][0] + tmp[i + 1][2]) or (tmp[i + 1][0] > tmp[i][0] and tmp[i + 1][0] < tmp[i][0] + tmp[i][2]):
    #         if tmp[i][4] == min(tmp[i][4], tmp[i + 1][4]):
    #             tmp.remove(tmp[i])
    #         else:
    #             tmp.remove(tmp[i + 1])
    number = ""
    if len(tmp) == 6 or len(tmp) == 7:
        for x in tmp: 
            number += decode(x[5])
        return number
    else:
        return "No signal"


while(True):
    ret, raw = cap.read()
    start = time.time()
    raw = cv2.putText(raw, str(int(fps)) + "FPS", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    try:
        results = model_plate(raw)

        #Results
        coordinates = results.xywh[0]

        for coordinate in coordinates:
            x, y, w, h = int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[3])

            # plate fetch
            plate = raw[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

            #draw contour
            raw = cv2.rectangle(raw, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
            raw = cv2.putText(raw, plate_detect(plate), (int(x - w / 2), int(y - h / 2) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        
        cv2.imshow("frame", raw)
    except IndexError:
        cv2.imshow("frame", raw)
        pass
    end = time.time()
    fps = 1 / (end - start)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()









# raw = cv2.resize(raw, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_AREA)

#     # 灰階化
# gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

# # 臨界二值化(>200:255, <200:0)
# ret, frame = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# #     # Edge Detect
# # edge = cv2.Canny(frame, 50, 254)

# # 將圖片進行重複腐蝕、膨脹
# for i in range(3):
#     kernel = np.ones((i + 2, i + 2),np.uint8)
#     frame = cv2.dilate(frame, kernel, iterations = 2)

#     kernel = np.ones((i + 1, i + 1),np.uint8)
#     frame = cv2.erode(frame, kernel, iterations = 2)
# cv2.imshow("frame", frame)

# # 尋找長方形輪廓
# cnts = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# i = 0
# for cnt in cnts:
#     x, y, w, h = cv2.boundingRect(cnt)
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
    
#     max = np.argmax(box, axis=0)
#     min = np.argmin(box, axis=0)
#     width = abs(box[min[0]][0] - box[max[0]][0])
#     height = abs(box[min[1]][1] - box[max[1]][1])
    
#     if width/height > 1.75 and width/height < 2.75 and height > 10:
#         i += 1
#         mask = raw[box[min[1]][1]:box[max[1]][1], box[min[0]][0]:box[max[0]][0]]
#         plate_number = plate.plate_detect(mask)
#         mix = cv2.drawContours(raw,[box], 0, (0, 0, 255), 1)
#         mix = cv2.putText(mix, plate_number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#         frame = cv2.drawContours(frame,[box], 0, (0, 0, 255), 1)
#         frame = cv2.putText(frame, plate_number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


# print(time.time() - c)
# cv2.imshow("frame", frame)
# cv2.imshow("mask", frame)
# cv2.imshow("gray", mix)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

    
#釋放攝影機
# cap.release()

#關閉所有 OpenCV 視窗
# cv2.destroyAllWindows()