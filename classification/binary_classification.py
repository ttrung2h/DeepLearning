import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np
class ClassificationModel(nn.Module):
    def __init__(self,num_in,num_out) :
        super().__init__()
        self.hiddenlayer1 = num_in*20
        self.hiddenlayer2 = num_in*20
        self.layer1 = nn.Linear(in_features=num_in,out_features= self.hiddenlayer1)
        self.layer2 = nn.Linear(in_features=self.hiddenlayer1,out_features=self.hiddenlayer2)
        self.layer3 = nn.Linear(in_features=self.hiddenlayer2,out_features=num_out)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
    def caculateAccurracy(self,y_true,y_pre):
        correct = torch.eq(y_true,y_pre).sum().item()
        result = (correct/len(y_pre))*100
        return result

# Create circles data
n_samples = 1000
X,y = make_circles(n_samples=n_samples,noise=0.09,random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)


# Change data to tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))



# Setting optimizer and loss function

model = ClassificationModel(2,1)
optimizer = torch.optim.SGD(params=model.parameters(),lr= 0.1)
loss_fn = nn.BCEWithLogitsLoss()

# tranning loop
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')



X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

torch.manual_seed(123)
epochs = 1000
for epoch in range(epochs):
    
    model.train()
    
    # Predict 
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # accuracy
    accuracy = model.caculateAccurracy(y_train,y_pred)
    # caculate loss
    loss = loss_fn(y_logits,y_train)
    # zero grad
    optimizer.zero_grad()
    # back propagation
    loss.backward()
    # optimize step
    optimizer.step()
    
    # testing
    model.eval()

    with torch.no_grad():
        if epoch%100 == 0:
            y_logits_test = model(X_test).squeeze()
            y_pred_test = torch.round(torch.sigmoid(y_logits_test))
            # accuracy
            accuracy_test = model.caculateAccurracy(y_test,y_pred_test)
            # caculate loss
            loss_test = loss_fn(y_logits_test,y_test)
            print(f"Epoch: {epoch}|Loss: {loss}|Accuracy: {accuracy}|Loss_test: {loss_test}|Accuracy test :{accuracy_test}%")

