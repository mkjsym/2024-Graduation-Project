import cv2
import mediapipe as mp
import numpy as np
import torch.nn as nn
import torch
from time import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(GRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


sequence_length = 30
input_size = 132
num_layers = 2
hidden_size = 50
model = GRU(input_size = input_size, hidden_size = hidden_size, sequence_length = sequence_length, num_layers = num_layers, device = device).to(device)
model.load_state_dict(torch.load(f=r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-Graduation-Project/Sources/Data/Weights/weight.pth'))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence = 0.1, model_complexity = 2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-Graduation-Project/Sources/Data/video (3).avi')

queue_size = 30
queue = []
flag = 0

# Initialize a variable to store the time of the previous frame.
time1 = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No Frame")
        break

    #image.flags.writeable = False
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image2)

    x = []
    try:
        for k in range(33):
            x.append(result.pose_landmarks.landmark[k].x)
            x.append(result.pose_landmarks.landmark[k].y)
            x.append(result.pose_landmarks.landmark[k].z)
            x.append(result.pose_landmarks.landmark[k].visibility)
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
        cv2.putText(image, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2

    if (flag == 1):
        cv2.putText(image, 'Warning!', (10, 65),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, 'Hello World', (10, 65),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    if len(queue) == queue_size:
        input = torch.FloatTensor(queue).to(device)
        input = input.unsqueeze(0)
        out = model(input)
        if out > 2.5:
            flag = 1
        else:
            flag = 0
        print(out, flag)
        queue.pop(0)

    #image.flags.writeable = True
    cv2.imshow('Webcam', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
