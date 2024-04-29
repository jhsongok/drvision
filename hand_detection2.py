import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1

gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# MediaPipe Hands 손의 관절 위치를 인식할 수 있는 모델을 초기화한다.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=max_num_hands,   # 최대 손의 인식 갯수
    min_detection_confidence=0.5,  # 탐지 임계치
    min_tracking_confidence=0.5    # 추적 임계치
)

file = np.genfromtxt('gesture_train.csv', delimiter=',') # 손가락과 손가락 사이의 각도, 제스추어(분류) 파일을 읽어온다.
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

knn = cv2.ml.KNearest_create()  # KNN 모델을 초기화
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # KNN 모델 학습

while cap.isOpened():
    ret, img  = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))

            for j, lm in enumerate(res.landmark): # lm 은 0 ~ 1 사이의 상대좌표를 가지고 있다.
                joint[j] = [lm.x, lm.y, lm.z]

            # 관절 사이의 각도를 계산한다.
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint

            v = v2 - v1 # (20, 3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.
            v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1)  # (20, 3) / (20, 1) 유닛 벡터를 구한다. (벡터 / 벡터의 길이)
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,] 유닛 벡터를 내적한 값의 아크코사인을 구하면 관절 사이의 각도를 구할 수 있다.
            
            angle = np.degrees(angle) # Convert radian to degree
            angle = np.expand_dims(angle.astype(np.float32), axis=0) # 머신러닝 모델에 넣어서 추론할때는 항상 맨앞 차원 하나를 추가한다.

            # 제스처를 추론한다.
            _, results, _, _ =  knn.findNearest(angle, 3)  # k = 3
            
            idx = int(results[0][0])  # 인덱스 정수 형태로 변환(0 ~ 10까지 총 11가지로 분류)
     
            if idx in rps_gesture.keys():  # 가위, 바위, 보 중에 하나면 출력하기
                gesture_name = rps_gesture[idx]

                cv2.putText(img, text=gesture_name.upper(), org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 0, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', resized)        
    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
