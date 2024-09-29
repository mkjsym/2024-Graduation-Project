ffmpeg 라이브러리를 통한 영상 송신
1. ffmpeg 라이브러리 설치
    sudo apt-get update
    sudo apt-get install ffmpeg
2. rtsp 서버 라이브러리 설치
    mediamtx 깃허브 접속
    rtsp 웹서버 엔진 실행 -> ./mediamtx
3. ffmpeg를 통한 실시간 영상 송신
    ffmpeg -i /dev/video0 -vcodec libx264 -acodec aac -f rtsp rtsp://192.168.25.128:8554/mystream
4. 로컬 컴퓨터에서 파이썬 코드를 통한 영상 수신