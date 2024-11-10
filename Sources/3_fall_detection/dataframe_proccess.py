import pandas as pd
import random
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


sequence_length = 30
split = 9734 #train test split point 7 : 3
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
