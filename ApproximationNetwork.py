import torch

class RegressionNet(torch.nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.il = torch.nn.Linear(1, 20)
        self.af1 = torch.nn.Sigmoid()
        self.hl = torch.nn.Linear(20, 50)
        self.af2 = torch.nn.ReLU()
        self.ol = torch.nn.Linear(50, 1)
    
    
    def forward(self, x):
        x = self.il(x)
        x = self.af1(x)
        x = self.hl(x)
        x = self.af2(x)
        x = self.ol(x)
        return x
    
    
    
net = RegressionNet()

def target_function(x):
    return 2**x * torch.sin(2**-x)
    
#Подготовка датасета
x_train =  torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

def metric(pred, target):
    return (pred - target).abs().mean()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

def loss(pred, target):
    return ((pred-target)**2).mean()
                             
for epoch_index in range(500):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()

print(metric(net.forward(x_validation), y_validation).item())
