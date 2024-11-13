import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
import cvzone
import cv2
import math
import datetime
import sys
import mediapipe as mp
import numpy as np
import torch.nn as nn
import torch
from time import time

def sendMessage(title, content):
    #use port 587 or 465
    smtp = smtplib.SMTP('smtp.gmail.com', 587)

    smtp.ehlo()
    smtp.starttls()

    #sender email, sender's app password
    smtp.login(senderEmail, senderPW)

    msg = MIMEText(content)
    msg['Subject'] = title

    smtp.sendmail(senderEmail, receiverEmail, msg.as_string())

    smtp.quit()

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
