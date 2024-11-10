# extract human skeletons and make labels per frame.
# fine-tuned for Le2i Datasets.
# Writer: YoungMin Jeon

import cv2
import os
import pandas as pd
import natsort
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.1, model_complexity = 2)
mp_drawing = mp.solutions.drawing_utils

def label_frames(video_dir, annotation_dir, output_directory):
    """
    영상 데이터셋에 레이블을 부여하고 CSV 파일로 저장하는 함수

    Args:
        video_dir: 영상 파일이 저장된 디렉토리
        annotation_dir: annotation 파일이 저장된 디렉토리
        output_csv: 레이블 정보를 저장할 CSV 파일 경로
    """

    df = pd.DataFrame()
    for video_file in natsort.natsorted(os.listdir(video_dir)):
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

            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_img)
            x = []

            if not result.pose_landmarks:
                frame_count += 1
                continue

            try:
                for k in range(33):
                    x.append(result.pose_landmarks.landmark[k].x)
                    x.append(result.pose_landmarks.landmark[k].y)
                    x.append(result.pose_landmarks.landmark[k].z)
                    x.append(result.pose_landmarks.landmark[k].visibility)
            except AttributeError:
                x = np.zeros(132)
            
            # 레이블 부여
            label = 0  # 정상
            if start_frame <= frame_count <= end_frame:
                label = 1  # 넘어짐
            
            x.append(label)

            tmp = pd.DataFrame(x).T
            df = pd.concat([df, tmp])

            print(video_file, frame_count)
            frame_count += 1

        cap.release()

    # DataFrame 생성 및 CSV 파일 저장
    df.to_csv(output_directory + '/skeleton_labels_' + 'Coffee_room_01-revised' + '.csv', index=False)

# 사용 예시
video_directory = r"/home/youngmin/YM/SL_disk_b/datasets/Le2i/Coffee_room_01/Coffee_room_01/Videos/"
annotation_directory = r"/home/youngmin/YM/SL_disk_b/datasets/Le2i/Coffee_room_01/Coffee_room_01/Annotation_files/"
output_directory = r"/home/youngmin/YM/Github/2024-Graduation-Project/Sources/Data/Action_Labels/"
label_frames(video_directory, annotation_directory, output_directory)
