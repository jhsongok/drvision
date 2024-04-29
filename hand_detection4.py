from cvzone.HandTrackingModule import HandDetector
import cv2

detector = HandDetector(maxHands=1)

cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_video = cv2.VideoCapture('Video.mp4')

w = int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # cap_cam의 width(가로) 길이를 가져옴
total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT)) # cap_video의 전체 Frame 수를 가져옴
print(total_frames) # 326개

_, video_img = cap_video.read()  # 첫번째 프레임을 읽는다. 정지 상태

# 타임라인 함수 정의
def draw_timeline(video_img, rel_x):
    img_h, img_w = video_img.shape[:2]
    timeline_w = int(img_w * rel_x)
    cv2.rectangle(video_img, pt1=(0, img_h-50), pt2=(timeline_w, img_h-47), color=(0, 0, 255), thickness=-1)

# 타임라인과 변수 초기화
rel_x = 0
frame_idx = 0
draw_timeline(video_img, rel_x)

while cap_cam.isOpened():
    ret, cam_img = cap_cam.read()  # 카메라 프레임을 읽는다.
    if not ret:
        break

    cam_img = cv2.flip(cam_img, 1)  # 거울 반전

    hands, cam_img = detector.findHands(cam_img)  # 손의 랜드마크를 찾는다.

    if hands: # 카메라에서 손이 인식되면
        lm_list = hands[0]['lmList']  # 손의 랜드마크 리스트를 받는다.
        fingers = detector.fingersUp(hands[0]) # 손가락을 접으면 0, 펴면 1 (5개를 원소로 하는 리스트로 받는다.)

        #cv2.putText(cam_img, text=str(fingers), org=(50,120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)

        length, info, cam_img = detector.findDistance(lm_list[4][:2], lm_list[8][:2], cam_img)  # 엄지(lm_list[4])와 검지(lm_list[8]) 사이의 거리를 계산한다.
        
        if fingers == [0, 0, 0, 0, 0]:  # 정지 모드, 주먹을 쥐고 있을 때
            pass
        else: # 탐색 또는 플레이 모드
            if length < 30: # Navigate(탐색) 모드
                rel_x = lm_list[4][0] / w  # 엄지 손가락의 x좌표를 0~1 사이 값으로(상대좌표) 변환
                                           # lm_list[4][0] 은 엄지 손가락의 x 좌표, w 로 나눠주면 상대좌표가 됨
                frame_idx = int(rel_x * total_frames) # 이동하고자 하는 프레임 번호(frame_idx) 계산
                                                      # 엄지 손가락 x좌표에 따른 동영상 프레임 번호 계산
                if frame_idx < 0:
                    frame_idx = 0
                elif frame_idx > total_frames:
                    frame_idx = total_frames

                cap_video.set(1, frame_idx) # 동영상을 해당 프레임 인덱스(frame_idx)로 이동
                
                cv2.putText(cam_img, text='Navigate %.2f, %d' % (rel_x, frame_idx), org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)  
            else: # play(재생) 모드
                frame_idx = frame_idx + 1  # 재생 중임으로 프레임을 하나 더 한다.(다음 프레임 재생)
                rel_x = frame_idx / total_frames

            if frame_idx < total_frames:
                _, video_img = cap_video.read() # 동영상의 프레임을 읽는다.
                draw_timeline(video_img, rel_x) # 타임라인 업데이트

    cv2.imshow('cam', cam_img)
    cv2.imshow('video', video_img)
    if cv2.waitKey(5) & 0xFF == 27:
      break
