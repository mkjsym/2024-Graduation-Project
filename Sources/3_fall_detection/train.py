import mediapipe as mp
import os
import cv2
import pandas as pd
import natsort
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.1, model_complexity = 2)
mp_drawing = mp.solutions.drawing_utils

def PointExtractor(route):
        count = 0
        for r in os.listdir(route):
            print(count)
            df = pd.DataFrame()

            tmp_path = os.listdir(route + "/" + r)
            tmp_path = natsort.natsorted(tmp_path)
            for i in tmp_path[-90:]:
                img = cv2.imread(route + "/" + r + "/" + i)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_img)
                x = []
                try:
                    for k in range(33):
                        x.append(result.pose_landmarks.landmark[k].x)
                        x.append(result.pose_landmarks.landmark[k].y)
                        x.append(result.pose_landmarks.landmark[k].z)
                        x.append(result.pose_landmarks.landmark[k].visibility)
                except AttributeError:
                    x = np.zeros(132)
                tmp = pd.DataFrame(x).T
                df = pd.concat([df, tmp])

            df.to_csv('test%d.csv' % count)
            count += 1

if __name__ == '__main__':
    for j in range(1,13):
        num = j
        path = r"/home/youngmin/disk_b/datasets/HumanPose(Video)/sit/3-%d" % num
        destination = r"/home/youngmin/YM/My/csv/3-%d" % num
        if not os.path.exists(destination):
            os.makedirs(destination)
        os.chdir(destination)
        PointExtractor(path)
