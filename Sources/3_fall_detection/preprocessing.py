import cv2
import os
import pandas as pd

def label_frames(video_dir, annotation_dir, output_directory):
    """
    영상 데이터셋에 레이블을 부여하고 CSV 파일로 저장하는 함수

    Args:
        video_dir: 영상 파일이 저장된 디렉토리
        annotation_dir: annotation 파일이 저장된 디렉토리
        output_csv: 레이블 정보를 저장할 CSV 파일 경로
    """

    for video_file in os.listdir(video_dir):
        data = []
        video_path = os.path.join(video_dir, video_file)
        annotation_path = os.path.join(annotation_dir, video_file.replace(".avi", ".txt"))

        # 동영상 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)

        # annotation 파일 읽기
        with open(annotation_path, 'r') as f:
            start_frame = int(f.readline().strip())
            end_frame = int(f.readline().strip())

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 레이블 부여
            label = 0  # 정상
            if start_frame <= frame_count <= end_frame:
                label = 1  # 넘어짐

            data.append([video_file, frame_count, label])

            frame_count += 1

        cap.release()

        # DataFrame 생성 및 CSV 파일 저장
        df = pd.DataFrame(data, columns=['video_name', 'frame_number', 'label'])
        df.to_csv(output_directory + '/' + video_file + '_labels.csv', index=False)

# 사용 예시
video_directory = r"/home/youngmin/YM/SL_disk_b/datasets/Le2i/Coffee_room_01/Coffee_room_01/Videos/"
annotation_directory = r"/home/youngmin/YM/SL_disk_b/datasets/Le2i/Coffee_room_01/Coffee_room_01/Annotation_files/"
output_directory = r"/home/youngmin/YM/Github/2024-Graduation-Project/Sources/Data/Action_Labels/"
label_frames(video_directory, annotation_directory, output_directory)
