import cv2
import numpy as np
from matplotlib import pyplot as plt

# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return cnts


# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'
# Ham fine tune bien so, loai bo cac ki tu khong hop ly

def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')
plate_info = ""

img= cv2.imread('err3\er150a.jpg',1)
img = cv2.resize(src=img, dsize=(800, 600))

imgContour = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 50, 100])
upper_red1 = np.array([5, 255, 255])

lower_red2 = np.array([160, 50, 100])
upper_red2 = np.array([179, 255, 255])  

lower_red3 = np.array([110, 169, 158])
upper_red3 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask3 = cv2.inRange(hsv, lower_red3, upper_red3)

maskk = cv2.bitwise_or(mask1, mask2)
mask = cv2.bitwise_or(maskk, mask3)

#cv2.imshow('mask',mask)

result = cv2.bitwise_and(img, img, mask = mask)
mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
hStack = np.hstack([img, mask])

imgBlur= cv2.GaussianBlur(result,(9,9),1)
imgGray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.uint8) #ma tran 5x5
imgCanny = cv2.Canny(imgGray, 30, 200)
imgDil= cv2.dilate(imgCanny,kernel,iterations=1)
#cv2.imshow('qweqwe',imgDil)
contours, _ = cv2.findContours(imgDil,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    

cood_xmax = 0
cood_ymax = 0
cood_xmin = 0
cood_ymin = 0
dem = 0
khoang = 30
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:3]

img1 = imgContour.copy()
for cnt in contours:
    peri=cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,0.06*peri,True)
    #print(len(approx))
    x,y,w,h=cv2.boundingRect(approx)
    cv2.rectangle(img1,(x-3,y-3),(x+w+7,y+h+7),(0,255,0),2)
cv2.imshow('ferbwverf',img1)
    #cv2.drawContours(imgContour, contours,-1,(0, 255, 0),2)
    #cv2.imshow('contour1',imgContour)
    #Xac dinh vien xanh la cay
#===========================================================================
for cnt in contours:
    if dem == 0:
        peri=cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,0.06*peri,True)
        #print(len(approx))
        x,y,w,h=cv2.boundingRect(approx)
        cood_xmax = x + w
        cood_ymax = y + h
        cood_xmin = x
        cood_ymin = y
        cv2.rectangle(imgContour,(x-3,y-3),(x+w+7,y+h+7),(0,255,0),2)
        dem = 1
    else:
        peri=cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,0.06*peri,True)
        x,y,w,h=cv2.boundingRect(approx)
        if x > cood_xmax:
            #print(x - cood_xmax)
            if y > cood_ymin:
                if x - cood_xmax < khoang:
                    if y - cood_ymin < khoang:
                        cv2.rectangle(imgContour,(x-1,y-1),(x+w+2,y+h+2),(0,255,0),2)
                        cood_xmax = x + w
                        if cood_ymax < y+h:
                            cood_ymax = y + h
            else:
                if x - cood_xmax < khoang:
                    if cood_ymin - y < khoang:
                        cv2.rectangle(imgContour,(x-1,y-1),(x+w+2,y+h+2),(0,255,0),2)
                        cood_xmax = x + w
                        cood_ymin = y
                        if cood_ymax < y+h:
                            cood_ymax = y + h
        else:
            print(cood_xmin - x - w)
            print(cood_xmin, x, y, h, w)
            
            if y > cood_ymin:
                if (cood_xmin - x - w) < khoang:
                    if y - cood_ymin < khoang:
                        cv2.rectangle(imgContour,(x-1,y-1),(x+w+2,y+h+2),(0,255,0),2)
                        cood_xmin = x
                        if cood_ymax < y+h:
                            cood_ymax = y + h
            else:
                if (cood_xmax - x - w) < khoang:
                    if cood_ymin - y < khoang:
                        cv2.rectangle(imgContour,(x-1,y-1),(x+w+2,y+h+2),(0,255,0),2)
                        cood_xmin = x
                        cood_ymin = y
                        if cood_ymax < y+h:
                            cood_ymax = y + h
    #xac dinh thong tin
    #cv.putText()
#======================================================================
roi = img[cood_ymin-5:cood_ymax+5, cood_xmin-5:cood_xmax + 5]

mask = maskk[cood_ymin-5:cood_ymax+5, cood_xmin-5:cood_xmax + 5]
imgContour1 = roi.copy()

#cv2.imshow('mask1',roi)
#cv2.imshow('mask',mask)
#==========================================================
Gray = cv2.cvtColor( imgContour1, cv2.COLOR_BGR2GRAY)
imgThre = cv2.threshold(Gray, 165, 255,cv2.THRESH_BINARY_INV)[1]

kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thre_mor = cv2.morphologyEx(imgThre, cv2.MORPH_DILATE, kernel3)
cv2.imshow('aaa',thre_mor)
#cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#==========================================================
res = cv2.bitwise_and(roi, roi, mask = mask)
mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
hStack = np.hstack([roi, mask])
imgBlur= cv2.GaussianBlur(res,(9,9),1)
imgGray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)
kernel = np.ones((3,3),np.uint8) #ma tran 5x5
imgCanny = cv2.Canny(imgGray, 30, 200)
imgDil= cv2.dilate(imgCanny,kernel,iterations=1)

#cv2.imshow('aaa',imgCanny)

contours1, _ = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

plate_info = ""

#=====================================================
# Cau hinh tham so cho model SVM

for c in sort_contours(contours1):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 0.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
        if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

            # Ve khung chu nhat quanh so
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Tach so va predict
            curr_num = thre_mor[y:y+h,x:x+w]
            #cv2.imshow('qqqq',curr_num)
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))            
            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
            curr_num = np.array(curr_num,dtype=np.float32)
            
            curr_num = curr_num.reshape(1, digit_w * digit_h)
            #cv2.imshow('v1',curr_num)
            # Dua vao model SVM
            result = model_svm.predict(curr_num)[1]
            result = int(result[0, 0])
            #cv2.imshow('v2',result)
            if result<=9: # Neu la so thi hien thi luon
                result = str(result)
            else: #Neu la chu thi chuyen bang ASCII
                result = chr(result)

            plate_info +=result

    # Viet bien so len anh
cv2.putText(img,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255),5, lineType=cv2.LINE_AA)
#print(plate_info)
if plate_info == "02":
    ima = cv2.imread('bando02.PNG', 1)
    cv2.imshow('duong di xe 02', ima)
    # cv2.imshow('0', roi)
    cv2.imshow('1', img)
    # cv2.imshow('2', imgContour)
    cv2.waitKey()
    plt.show
    # cv2.waitKey()
elif plate_info == "19":
    ima = cv2.imread('bando19.PNG', 1)
    cv2.imshow('duong di xe 19', ima)
    # cv2.imshow('0', roi)
    cv2.imshow('1', img)
    # cv2.imshow('2', imgContour)
    cv2.waitKey()
    plt.show
    # cv2.waitKey()
elif plate_info == "36":
    ima = cv2.imread('bando36.PNG', 1)
    cv2.imshow('duong di xe 36', ima)
    # cv2.imshow('0', roi)
    cv2.imshow('1', img)
    # cv2.imshow('2', imgContour)
    cv2.waitKey()
    plt.show
    # cv2.waitKey()
elif plate_info == "55":
    ima = cv2.imread('bando55.PNG',1)
    cv2.imshow('duong di xe 55', ima)
    #cv2.imshow('0', roi)
    cv2.imshow('1', img)
    #cv2.imshow('2', imgContour)
    cv2.waitKey()
    plt.show
    #cv2.waitKey()
elif plate_info == "150":
    ima = cv2.imread('bando150.PNG',1)
    cv2.imshow('duong di xe 150', ima)
    #cv2.imshow('0', roi)
    cv2.imshow('1', img)
    #cv2.imshow('2', imgContour)
    cv2.waitKey()
    plt.show
    #cv2.waitKey()
#cv2.destroyAllWindows()
#===============================================================
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#imgContour1 = cv2.cvtColor(imgContour, cv2.COLOR_BGR2RGB)
#cv2.imshow('0',roi)
#cv2.imshow('1',img)
#cv2.imshow('2',imgContour)
#cv2.waitKey()
#plt.show
