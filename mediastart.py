import cv2
import sys

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    frame = cv2.flip(frame, 1)  # 0: 상하대칭, 1 : 좌우대칭 

    cv2.imshow('Video display', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()