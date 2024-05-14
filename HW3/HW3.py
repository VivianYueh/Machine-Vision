import cv2
import cv2.aruco as aruco
import numpy as np

def augmentation(bbox, img, img_augment):
    top_left = bbox[0][0][0], bbox[0][0][1]
    top_right = bbox[0][1][0], bbox[0][1][1]
    bottom_right = bbox[0][2][0], bbox[0][2][1]
    bottom_left = bbox[0][3][0], bbox[0][3][1]
    '''print(top_left)
    print(top_right)
    print(bottom_right)
    print(bottom_left)'''
    height, width, _, = img_augment.shape

    points_1 = np.array([top_left, top_right, bottom_right, bottom_left])
    points_2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix, _ = cv2.findHomography(points_2, points_1)
    image_out = cv2.warpPerspective(img_augment, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, points_1.astype(int), (0, 0, 0))
    image_out = img + image_out

    return image_out

cap = cv2.VideoCapture('HW3/CharUco_board.mp4')
#原始畫面有點大，為了有利於顯示這份講義所以縮小。    
totalFrame   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
frameHeight  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

arucoParams  = aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

arucoDict    = aruco.Dictionary_get(aruco.DICT_6X6_250)

# 必須描述ChArUco board的尺寸規格
gridX        = 5 # 水平方向5格
gridY        = 7 # 垂直方向7格
squareSize   = 4 # 每格為4cmX4cm
# ArUco marker為2cmX2cm
charucoBoard = aruco.CharucoBoard_create(gridX,gridY,squareSize,squareSize/2,arucoDict)

print('height {}, width {}'.format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
refinedStrategy = True
criteria        = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
frameId        = 0
collectCorners = []
collectIds     = []
collectFrames  = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame,(frameWidth,frameHeight)) 
    (corners, ids, rejected) = aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)    
    
    if refinedStrategy:
        corners, ids, _, _ = aruco.refineDetectedMarkers(frame,charucoBoard,corners,ids,rejected)
        
    if frameId % 100 == 50 and ids is not None and len(ids)==17: # 17 ArUco markers
        collectCorners.append(corners)
        collectIds.append(ids.ravel())
        collectFrames.append(frame)
        
    if len(corners) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)
        

    #cv2.imshow('Analysis of a CharUco board for camera calibration',frame)
    if cv2.waitKey(20) != -1:
        break
        
    frameId += 1

caliCorners=np.concatenate([np.array(x).reshape(-1,4,2) for x in collectCorners],axis=0)
counter=np.array([len(x) for x in collectIds])
caliIds=np.array(collectIds).ravel()
cameraMatrixInit = np.array([[ 1000.,    0., frameWidth/2.],[    0., 1000., frameHeight/2.],[    0.,    0.,           1.]])
distCoeffsInit   = np.zeros((5,1))
ret, aruco_cameraMatrix, aruco_distCoeffs, aruco_rvects, aruco_tvects = aruco.calibrateCameraAruco(caliCorners,caliIds,counter,charucoBoard,(frameWidth,frameHeight),cameraMatrixInit,distCoeffsInit)
print(aruco_cameraMatrix)
print(aruco_distCoeffs)

caliCorners=[]
caliIds    =[]
for corners, ids, frame in zip(collectCorners,collectIds,collectFrames):
    ret, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners,ids,frame,charucoBoard,aruco_cameraMatrix,aruco_distCoeffs)
    caliCorners.append(charucoCorners)
    caliIds.append(charucoIds)

ret, charuco_cameraMatrix, charuco_distCoeffs, charuco_rvects, charuco_tvects = aruco.calibrateCameraCharuco(caliCorners,caliIds,charucoBoard,(frameWidth,frameHeight), aruco_cameraMatrix,aruco_distCoeffs)    
print(charuco_cameraMatrix)
print(charuco_distCoeffs)

#cv2.destroyAllWindows()
cap.release()

#im_src = cv2.imread('HW3/test2.jpg')
v1=cv2.VideoCapture('HW3/hw3_1.mp4')
v2=cv2.VideoCapture('HW3/hw3_2.mp4')
v3=cv2.VideoCapture('HW3/hw3_3.mp4')
v4=cv2.VideoCapture('HW3/hw3_4.mp4')
v5=cv2.VideoCapture('HW3/hw3_5.mp4')
v6=cv2.VideoCapture('HW3/hw3_6.mp4')

ret1, frame1 = v1.read()
ret2, frame2 = v2.read()
ret3, frame3 = v3.read()
ret4, frame4 = v4.read()
ret5, frame5 = v5.read()
ret6, frame6 = v6.read()

detection1 = False
frame_count1 = 0
detection2 = False
frame_count2 = 0
detection3 = False
frame_count3 = 0
detection4 = False
frame_count4 = 0
detection5 = False
frame_count5 = 0
detection6 = False
frame_count6 = 0

cap = cv2.VideoCapture('HW3/arUco_marker.mp4')

#原始畫面有點大，為了有利於顯示這份講義所以縮小。    
frameWidth   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
frameHeight  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

arucoParams = aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
arucoDict   = aruco.Dictionary_get(aruco.DICT_7X7_50)

#print('height {}, width {}'.format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FRAME_WIDTH)))


while True:
    ret, frame = cap.read()
    '''ret1, frame1 = v1.read()
    ret2, frame2 = v2.read()
    ret3, frame3 = v3.read()
    ret4, frame4 = v4.read()
    ret5, frame5 = v5.read()
    ret6, frame6 = v6.read()'''
    if not ret:
        break
    if detection1 == False:
        v1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count1 = 0
    else:
        if frame_count1 == v1.get(cv2.CAP_PROP_FRAME_COUNT):
            v1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count1 = 0
        ret1, frame1 = v1.read()
    if detection2 == False:
        v2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count2 = 0
    else:
        if frame_count2 == v2.get(cv2.CAP_PROP_FRAME_COUNT):
            v2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count2 = 0
        ret2, frame2 = v2.read()
    if detection3 == False:
        v3.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count3 = 0
    else:
        if frame_count3 == v3.get(cv2.CAP_PROP_FRAME_COUNT):
            v3.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count3 = 0
        ret3, frame3 = v3.read()
    if detection4 == False:
        v4.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count4 = 0
    else:
        if frame_count4 == v4.get(cv2.CAP_PROP_FRAME_COUNT):
            v4.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count4 = 0
        ret4, frame4 = v4.read()
    if detection5 == False:
        v5.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count5 = 0
    else:
        if frame_count5 == v5.get(cv2.CAP_PROP_FRAME_COUNT):
            v5.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count5 = 0
        ret5, frame5 = v5.read()
    if detection6 == False:
        v6.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count6 = 0
    else:
        if frame_count6 == v6.get(cv2.CAP_PROP_FRAME_COUNT):
            v6.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count6 = 0
        ret6, frame6 = v6.read()
    
    frame = cv2.resize(frame,(frameWidth,frameHeight)) 
    frame = cv2.undistort(frame, charuco_cameraMatrix, charuco_distCoeffs)
    frame1=cv2.flip(frame1,0)
    frame1=cv2.flip(frame1,1)
    frame2=cv2.flip(frame2,0)
    frame2=cv2.flip(frame2,1)
    frame3=cv2.flip(frame3,0)
    frame3=cv2.flip(frame3,1)
    frame4=cv2.flip(frame4,0)
    frame4=cv2.flip(frame4,1)
    frame5=cv2.flip(frame5,0)
    frame5=cv2.flip(frame5,1)
    frame6=cv2.flip(frame6,0)
    frame6=cv2.flip(frame6,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (corners, ids, rejected) = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
    if len(corners) > 0:  
        #aruco.drawDetectedMarkers(frame, corners, ids)
        #print(corners)
        if ids is not None:    
            i=0
            for id in ids:        
                if id==1:
                    detection1=True
                    frame = augmentation(np.array(corners)[i], frame, frame1)
                elif id==2:
                    detection2=True
                    frame = augmentation(np.array(corners)[i], frame, frame2)
                elif id==3:
                    detection3=True
                    frame = augmentation(np.array(corners)[i], frame, frame3)
                elif id==4:
                    detection4=True
                    frame = augmentation(np.array(corners)[i], frame, frame4)
                elif id==5:
                    detection5=True
                    frame = augmentation(np.array(corners)[i], frame, frame5)
                elif id==6:
                    detection6=True
                    frame = augmentation(np.array(corners)[i], frame, frame6)
                i+=1
                
            
        cv2.imshow('result',frame)
        if cv2.waitKey(20) != -1:
            break
        frame_count1+=1
        frame_count2+=1
        frame_count3+=1
        frame_count4+=1
        frame_count5+=1
        frame_count6+=1
        
        
cv2.destroyAllWindows()
cap.release()
