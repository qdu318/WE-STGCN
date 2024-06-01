import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import torch.nn as nn
import torch.nn.functional as F

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 2)*dilation,
                              dilation=dilation)
        self.weight_norm = nn.utils.weight_norm(self.conv)
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1,padding=0)

    def forward(self, x1):
        #print("x1111",x1.shape[2])
        r=x1.shape[0]
        r2=x1.shape[2]
        x=x1.view(x1.shape[0],1,x1.shape[2])
        #print("Xxx",x.shape)
        #x=x.reshape(x.shape[1],1,x.shape[0])
        residual = x
        #print("residual",residual.shape)
        out = self.conv(x)
        #print("x", out.shape)
        out = self.weight_norm(x)
        out = self.dropout(out)
        out = self.tanh(out)
        #print("out",out.shape)
        if residual.shape[1] != out.shape[1]:
            residual = self.residual_conv(residual)
        #print("residual", residual.shape)
        out = out + residual
        return out

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, adj):
        row, col = adj.nonzero(as_tuple=True)
        col, row = col.flip(0), row.flip(0)
        row = torch.cat([row, row], dim=0)
        col = torch.cat([col, col], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        #print(edge_index.shape)
        h = self.conv1(x,edge_index)
        h=h.tanh()
        return h.T


class FeatureComponent(nn.Module):
    def __init__(self, weather_vocab_size, week_vocab_size,
                 embedding_dim, output_dim):
        super(FeatureComponent, self).__init__()

        # Embedding layers for each feature
        self.weather_embedding = nn.Embedding(weather_vocab_size, embedding_dim)
        self.week_embedding = nn.Embedding(week_vocab_size, embedding_dim)

        # Linear layer to combine all embeddings
        self.fc = nn.Linear(embedding_dim * 2, output_dim)

    def forward(self, weather, week):
        # Apply embeddings
        weather_embedded = self.weather_embedding(weather)
        week_embedded = self.week_embedding(week)
        # Concatenate all embeddings
        combined = torch.cat((weather_embedded, week_embedded), dim=-1)
        # Pass through a fully connected layer
        output = self.fc(combined)
        return output


class ThreeLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(ThreeLayerGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:,  :])  # Get the output of the last time step
        return out


class AttentionalPoolingMechanism(nn.Module):
    def __init__(self, feature_dim, gru_output_dim, hidden_dim):
        super(AttentionalPoolingMechanism, self).__init__()
        self.linear = nn.Linear(feature_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, A, B):
        # Unsqueeze A
        A = A.unsqueeze(1)  # Assume A is of shape [batch_size, feature_dim]
        # Linear transformation and Tanh activation
        #print("A",A.shape)
        AP = self.linear(A)
        AP = self.tanh(AP)
        # Unsqueeze AP for batch matrix multiplication
        AP = AP.unsqueeze(3)  # AP shape: [batch_size, hidden_dim, 1, 1]
        # Permute B to match dimensions for bmm
        B = B.permute(0, 2, 1).unsqueeze(3)  # B shape: [batch_size, gru_output_dim, seq_length, 1]
        # Batch matrix multiplication
        R = torch.bmm(AP.squeeze(3), B.squeeze(3))  # R shape: [batch_size, hidden_dim, gru_output_dim]
        # Normalize R
        R = torch.exp(-R)  # Apply exp(-r) as in the diagram
        R = R / R.sum(dim=-1, keepdim=True)  # Normalize to range [0, 1]
        # Batch matrix multiplication for final output
        Con = torch.bmm(R, B.squeeze(3).permute(0, 2, 1))  # Con shape: [batch_size, hidden_dim, seq_length]

        return Con.squeeze(2)  # Remove the singleton dimension


class CustomModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 假设输入维度为input_dim，输出维度为128
        self.bn1 = nn.BatchNorm2d(128)
        self.fc2 = nn.Linear(128, num_classes)  # 最后一层全连接，输出维度为num_classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x.unsqueeze(-1).unsqueeze(-1))  # BatchNorm2D 需要4D输入
        x = x.squeeze(-1).squeeze(-1)  # 去掉额外的维度
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class WE_STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_timesteps_output, num_nodes):
        super(WE_STGCN, self).__init__()
        self.temporal_conv_layer1 = TemporalConvLayer(in_channels, out_channels, kernel_size=3, dilation=1)
        self.temporal_conv_layer2 = TemporalConvLayer(out_channels, out_channels, kernel_size=3, dilation=2)
        self.temporal_conv_layer3 = TemporalConvLayer(out_channels, out_channels, kernel_size=3, dilation=4)
        self.graph_conv_layer = GCNLayer(num_timesteps_output, num_timesteps_output,K=5)
        self.featureComponent = FeatureComponent(num_timesteps_output,num_timesteps_output,num_timesteps_output,num_nodes)
        self.threeLayerGRU = ThreeLayerGRU(num_nodes*2,num_timesteps_output,num_timesteps_output,3)
        self.attentionalPoolingMechanism = AttentionalPoolingMechanism(num_nodes*2,num_nodes,num_timesteps_output)
        self.customModel = CustomModel(num_timesteps_output,num_nodes)

    def forward(self, x, adj,weather,week):
        x = x.squeeze(axis=0)
        #print("X",x.shape)
        x = x.reshape(x.shape[1],1,x.shape[0])

        x = self.temporal_conv_layer1(x)
        x = self.temporal_conv_layer2(x)
        x = self.temporal_conv_layer3(x)
        #print("TCN输出",x.shape)
        x=x.reshape(x.shape[0],x.shape[2])
        x = self.graph_conv_layer(x,adj)
        b = self.featureComponent(weather,week)
        #print("x+b",x.shape,b.shape)
        combined_features = torch.cat((x,b),dim=-1)
        #print("combined_features",combined_features.shape)
        GRUoutput = self.threeLayerGRU(combined_features)
        #print("GRU*3",GRUoutput.shape)
        GRUoutput = GRUoutput.reshape(GRUoutput.shape[1],1,GRUoutput.shape[0])

        output = self.attentionalPoolingMechanism(combined_features,GRUoutput)
        #print("APM",output.shape)
        output = output.reshape(output.shape[2], output.shape[0])
        output = self.customModel(output)
        #print("最后输出", output.shape)
        output = output.unsqueeze(0)
        return output

def train_model(model, train_loader, adj,weather,week,criterion ,optimizer,num_epochs=25):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        print("第", epoch, "轮次")
        for inputs, labels in train_loader:

            #print("input", inputs.shape)

            # 将梯度清零
            optimizer.zero_grad()
            #labels = labels.squeeze(axis=0)
            # 前向传播
            outputs = model(inputs,adj,weather,week)
            #print("outputs",outputs.shape)
            #print("标签大小", labels.shape)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            #print("反向传播前",inputs.shape)
            #inputs = inputs.squeeze(axis=0)
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            optimizer.step()

            # 统计损失
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Finished Training')

def evaluate_model(model, adj,weather,week,test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs,adj,weather,week)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    test_loss = running_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

def predict_future(model, future_data):
    model.eval()
    with torch.no_grad():
        future_data = future_data.to(device)
        predictions = model(future_data)
    return predictions

def dataload(path):

    data = pd.read_csv(path, header=None)
    data = data.iloc[:21600, :]
    # print(data.columns)
    data = data.sort_values(by=[data.columns[1], data.columns[0]])  # 按照第二列排序
    # print(data.head)
    # print(data.shape)
    third_column_data = data.iloc[:, 2].to_numpy().reshape(-1, 1)
    new_shape_data = third_column_data.reshape(-1, 30)
    # print("新数组",new_shape_data.shape)
    new_shape_data1 = new_shape_data[:672, :]
    # print(new_shape_data1[0][0])
    tensor_data = torch.tensor(new_shape_data1, dtype=torch.float32)
    reshaped_tensor = tensor_data.view(4, 168, 30)
    X = reshaped_tensor[:3, :, :]
    Y = reshaped_tensor[1:4, :, :]
    return X,Y


num_timesteps_input = 168  # 输入时间步长
num_timesteps_output = 168  # 输出时间步长
num_nodes = 30  # 节点数
in_channels = 1  # 输入通道数
out_channels = 1  # 输出通道数

model = WE_STGCN(in_channels=in_channels, out_channels=out_channels, num_timesteps_output=num_timesteps_output,
                 num_nodes=num_nodes)
adj = torch.randint(0, 2, (num_nodes, num_nodes)).float() # 随机生成邻接矩阵
x = torch.rand(num_timesteps_input, num_nodes)  # 随机生成输入数据 (batch_size, in_channels, num_timesteps_input)
weather = torch.tensor(np.random.randint(1, 4, size=168), dtype=torch.long)
week = torch.tensor(np.random.randint(1, 8, size=168), dtype=torch.long)
print(weather.shape,week.shape)

print(adj.shape)
out = model(x, adj,weather,week)
print("ss")
print(out.shape)  # 应输出 (batch_size, num_timesteps_output)

num_samples = 100  # 示例样本数
input_dim = (168, 30)

file_path = r'D:\Learn\data2\area_passenger_index.csv'


# 生成随机数据作为示例
#X = torch.randn(num_samples, *input_dim)
X,Y=dataload(file_path)

print(X.shape,Y.shape)

# 将数据包装成TensorDataset
dataset = TensorDataset(X, Y)

# 分割训练集和测试集，80%训练，20%测试
train_size = int(0.8 * len(dataset))

test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建DataLoader
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader,adj,weather,week, criterion, optimizer, num_epochs=3)
evaluate_model(model, adj,weather,week,test_loader, criterion)

#future_predictions = predict_future(model, future_data)

#torch.save(model.state_dict(), 'model_weights.pth')
#model.load_state_dict(torch.load('model_weights.pth'))