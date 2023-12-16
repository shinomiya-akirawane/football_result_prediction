import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from math import sin, cos, pi

def calculate_day_of_year(row):
    try:
        year = int(row['Year'])
        month = int(row['Month'])
        day = int(row['Day'])
        date = datetime(year, month, day)
        return date.timetuple().tm_yday
    except ValueError:
        return None

def add_trigonometric_date_features(df):
    df['day_of_year'] = df.apply(calculate_day_of_year, axis=1)
    df = df[df['day_of_year'].notna()]
    radians = (df['day_of_year'] / 365) * 2 * pi
    df['sin_date'] = np.sin(radians)
    df['cos_date'] = np.cos(radians)
    return df.drop(['Year', 'Month', 'Day', 'day_of_year'], axis=1)

train = pd.read_csv('final_train_data_date_unscaled.csv')
train = add_trigonometric_date_features(train)
x_train = train.drop('FTR', axis=1)
y_train = train['FTR']

test = pd.read_csv('final_test_data_date_unscaled.csv')
test = add_trigonometric_date_features(test)
x_test = test.drop('FTR', axis=1)
y_test = test['FTR']

x_train = torch.tensor(x_train.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

class FootballLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(FootballLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        lstm_out_last_step = lstm_out[-1]
        linear_out = self.linear1(lstm_out_last_step)
        activated_out = torch.relu(linear_out)
        predictions = self.linear2(activated_out)
        return predictions

model = FootballLSTM(input_dim=x_train.shape[1], hidden_dim=50)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(x_train, y_train.long())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(x_test, y_test.long())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    f1_test = evaluate_model(model, test_loader)
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Test F1: {f1_test:.2f}')

def plot_confusion_matrix(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)

    cm = confusion_matrix(y_test, predicted)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(model, x_test, y_test.long())

test_dataset = TensorDataset(x_test, y_test.long())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
f1_test = evaluate_model(model, test_loader)
print(f'Test F1 Score: {f1_test:.2f}')