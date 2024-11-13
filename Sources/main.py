from submodules import *


# sendMessage('title test', 'content test')


# 사전학습된 Yolo 모델 호출
model = YOLO('GitHub/2024-Graduation-Project/Sources/Data/fire_model.pt')
classnames = ['fire']

# Hyper Parameters
currentTime = datetime.datetime.now()
# Video_capture = cv2.VideoCapture(0)
# Video_capture = cv2.VideoCapture('video_path')
video_capture = cv2.VideoCapture('rtsp://192.168.0.226:8554/mystream')
# 웹캠 설정
video_capture.set(3, 640)  # 영상 가로길이 설정
video_capture.set(4, 480)  # 영상 세로길이 설정
fps = 30
streaming_window_width = int(video_capture.get(3))
streaming_window_height = int(video_capture.get(4))  

#현재 시간을 '년도 달 일 시간 분 초'로 가져와서 문자열로 생성
fileName = str(currentTime.strftime('%Y%m%d_%H%M%S'))
#파일 저장하기 위한 변수 선언
path = f'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/2024-Graduation-Project/Sources/Data/Videos/{fileName}.avi'

# DIVX 코덱 적용 # 코덱 종류 # DIVX, XVID, MJPG, X264, WMV1, WMV2
# 무료 라이선스의 이점이 있는 XVID를 사용
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# 비디오 저장
# cv2.VideoWriter(저장 위치, 코덱, 프레임, (가로, 세로))
out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))

# Hyper Parameters
senderEmail = 'mkjsym@gmail.com'
senderPW = 'itof hbrd duzh vwjw'
receiverEmail = 'mkjsym@gmail.com'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 30
input_size = 132
num_layers = 2
hidden_size = 50
model = GRU(input_size = input_size, hidden_size = hidden_size, sequence_length = sequence_length, num_layers = num_layers, device = device).to(device)
model.load_state_dict(torch.load(f=r'C:/Users/mkjsy/Desktop/YM/Source Code/GitHub/2024-Graduation-Project/Sources/Data/Weights/weight.pth'))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.1, model_complexity = 2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

queue_size = 30
queue = []
flag = 0

# Initialize a variable to store the time of the previous frame.
time1 = 0


while True:
    ret, frame = video_capture.read()
    # 촬영되는 영상보여준다. 프로그램 상태바 이름은 'streaming video' 로 뜬다.
    if frame is None:
        print('Image load failed')
        sys.exit()



    cv2.imshow('streaming video', frame)
    
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
