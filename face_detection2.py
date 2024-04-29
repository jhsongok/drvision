import cv2
import mediapipe as mp

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)  # 0: 상하대칭, 1 : 좌우대칭 
    h, w = img.shape[:2]
   
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = face_mesh.process(rgb_img)

    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * w)
            y = int(pt1.y * h)

            cv2.circle(img, (x, y), 1, (255, 255, 0), -1)
            #cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))

    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', resized)  
    #cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()