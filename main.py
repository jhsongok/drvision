import tensorflow.python.keras # tensorflow 2.8.x 에서는 tensorflow.keras 로 사용하면 에러
import numpy as np
import cv2

model = tensorflow.keras.models.load_model('keras_model.h5')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

classes = ['Scissors', 'Rock', 'Paper']

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    img = cv2.flip(img, 1)  # 0: 상하대칭, 1 : 좌우대칭 

    h, w = img.shape[:2]
    img = img[:, 160:160+h]  # 이미지를 정사각형으로 만들기
    img_input = cv2.resize(img, (224, 224)) # teachable machine 모델에 넣어주는 이미지 사이즈로 변경(224X224)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB) # BGR --> RGB
    img_input = (img_input.astype(np.float32) / 127.0) - 1.0  # -1 ~ 1 사이의 값으로 Feature scaling
    img_input = np.expand_dims(img_input, axis=0) # 0번째 축 차원을 늘려줌(1, 224, 224, 3)
    
    prediction = model.predict(img_input)
    idx = np.argmax(prediction)
    cv2.putText(img, text=classes[idx], org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 0, 255), thickness=2)
    #print(prediction)

    cv2.imshow('result', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
