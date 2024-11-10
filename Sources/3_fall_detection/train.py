import pandas as pd
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Hyper Parameters
num_layers = 2
hidden_size = 50
num_epochs = 700
split = 9734 #train test split point 7 : 3
sequence_length = 30
batch_size = 32

df = pd.read_csv(r"/home/youngmin/YM/Github/2024-Graduation-Project/Sources/Data/Action_Labels/skeleton_labels_Coffee_room_01-revised.csv")
save_path = r'/home/youngmin/YM/Github/2024-Graduation-Project/Sources/Data/Weights/weight.pth'
#print(df.head())


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

input_size = x_seq.size(2)

model = GRU(input_size = input_size, hidden_size = hidden_size, sequence_length = sequence_length, num_layers = num_layers, device = device).to(device)
#model.load_state_dict(torch.load(f=r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/SHM/My/savepoint.pth'))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

loss_graph = []
n = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        seq, target = data
        out = model(seq)
        loss = criterion(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    loss_graph.append(running_loss/n)
    if (epoch + 1) % 100 == 0:
        print('[epoch: %d] loss: %.4f' %(epoch, running_loss/n))

torch.save(obj=model.state_dict(), f=save_path)

# plt.figure(figsize=(20,10))
# plt.plot(loss_graph)
# plt.show() 
