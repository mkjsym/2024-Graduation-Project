from submodules import *

# 사전학습된 Yolo 모델 호출
fire_model = YOLO(r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-Graduation-Project/Sources/Data/fire_model.pt')
classnames = ['fire']

# Hyper Parameters
currentTime = datetime.datetime.now()
video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture('video_path')
# video_capture = cv2.VideoCapture('rtsp://192.168.0.226:8554/mystream')
# 웹캠 설정
video_capture.set(3, 640)  # 영상 가로길이 설정
video_capture.set(4, 480)  # 영상 세로길이 설정
fps = 30
streaming_window_width = int(video_capture.get(3))
streaming_window_height = int(video_capture.get(4))  

#현재 시간을 '년도 달 일 시간 분 초'로 가져와서 문자열로 생성
fileName = str(currentTime.strftime('%Y%m%d_%H%M%S'))
#파일 저장하기 위한 변수 선언
path = f'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-Graduation-Project/Sources/Data/Videos/{fileName}.avi'

# DIVX 코덱 적용 # 코덱 종류 # DIVX, XVID, MJPG, X264, WMV1, WMV2
# 무료 라이선스의 이점이 있는 XVID를 사용
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# 비디오 저장
# cv2.VideoWriter(저장 위치, 코덱, 프레임, (가로, 세로))
out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 30
input_size = 132
num_layers = 2
hidden_size = 50
pose_model = GRU(input_size = input_size, hidden_size = hidden_size, sequence_length = sequence_length, num_layers = num_layers, device = device).to(device)
pose_model.load_state_dict(torch.load(f=r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-Graduation-Project/Sources/Data/Weights/weight.pth'))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.1, model_complexity = 2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

queue_size = 30
queue = []
flag = 0

# Initialize a variable to store the time of the previous frame.
time1 = 0
fire_count = 0
fall_count = 0
count_threshold = 20


while True:
    success, frame = video_capture.read()
    # 촬영되는 영상보여준다. 프로그램 상태바 이름은 'streaming video' 로 뜬다.
    if (not success):
        print('failed to read video')
        continue

    fire_frame = cv2.resize(frame, (640, 480))
    fire_result = fire_model(frame, stream = True)

    pose_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(pose_frame)

    # Getting bbox,confidence and class names information to work with
    for info in fire_result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 65:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
                fire_count += 1
                
    x = []
    try:
        for k in range(33):
            x.append(pose_result.pose_landmarks.landmark[k].x)
            x.append(pose_result.pose_landmarks.landmark[k].y)
            x.append(pose_result.pose_landmarks.landmark[k].z)
            x.append(pose_result.pose_landmarks.landmark[k].visibility)
    except AttributeError:
        x = np.zeros(132)
    queue.append(x)
    
    # Set the time for this frame to the current time.
    time2 = time()
    
    # Check if the difference between the previous and this frame time &gt; 0 to avoid division by zero.
    if (time2 - time1) > 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2

    if len(queue) == queue_size:
        input = torch.FloatTensor(queue).to(device)
        input = input.unsqueeze(0)
        output = pose_model(input)
        if output > 2.5:
            flag = 1
            fall_count += 1
        else:
            flag = 0
        print(output, flag)
        queue.pop(0)

    if (flag == 1):
        cv2.putText(frame, 'Warning!', (10, 65),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, 'Hello World', (10, 65),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow('streaming video', frame)

    if (fire_count >= count_threshold):
        fire_count = 0
        curTime = datetime.datetime.now()
        timeText = str(curTime.strftime('%Y%m%d_%H%M%S'))
        sendMessage('Warning! Fire Detected', f'Warning. Fire Detected. {timeText}')
    
    if (fall_count >= count_threshold):
        fall_count = 0
        curTime = datetime.datetime.now()
        timeText = str(curTime.strftime('%Y%m%d_%H%M%S'))
        sendMessage('Warning! Fall Detected', f'Warning. Fall Detected. {timeText}')
    
    # 영상을 저장한다.
    out.write(frame)
    
    # 1ms뒤에 뒤에 코드 실행해준다.
    k = cv2.waitKey(1) & 0xff
    #키보드 esc 누르면 종료된다.
    if cv2.waitKey(5) & 0xFF == 27:
        break

video_capture.release()  # cap 객체 해제
out.release()  # out 객체 해제
cv2.destroyAllWindows()
