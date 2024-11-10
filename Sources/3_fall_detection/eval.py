import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Hyper Parameters
num_layers = 2
input_size = 132
hidden_size = 50
num_epochs = 700
split = 9734 #train test split point 7 : 3
sequence_length = 30
batch_size = 32

df = pd.read_csv(r"/home/youngmin/YM/Github/2024-Graduation-Project/Sources/Data/Action_Labels/skeleton_labels_Coffee_room_01-revised.csv")


def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    range_list = list(range(0, len(x) - sequence_length))
    random.shuffle(range_list)
    for i in range_list:
        x_seq.append(x[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1,1)

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
        out = out.reshape(out.shape[0], -1) # <- state 추가
        out = self.fc(out)
        return out


columns = range(0, 132)
X_columns = []
for i in columns:
    X_columns.append(str(i))
Y_column = '132'
X = df[X_columns].values
Y = df[Y_column].values
Y = Y.astype(int)

x_seq, y_seq = seq_data(X, Y, sequence_length)
x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)
train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size)

model = GRU(input_size = input_size, hidden_size = hidden_size, sequence_length = sequence_length, num_layers = num_layers, device = device).to(device)
model.load_state_dict(torch.load(f=r'/home/youngmin/YM/Github/2024-Graduation-Project/Sources/Data/Weights/weight.pth'))
# 모델 평가
model.eval()  # 모델을 평가 모드로 설정
with torch.no_grad():
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for seq, target in test_loader:
        out = model(seq)
        _, predicted = torch.max(out.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

# 혼동 행렬 출력
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['0', '1'], rotation=45)
plt.yticks(tick_marks, ['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
