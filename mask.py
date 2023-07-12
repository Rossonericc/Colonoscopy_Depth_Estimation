import torch
import torch.nn as nn
import torch.optim as optim


class MaskedLoss(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, y_pred, y_true):
        y_pred_masked = torch.masked_select(y_pred, self.mask)
        y_true_masked = torch.masked_select(y_true, self.mask)
        return nn.functional.mse_loss(y_pred_masked, y_true_masked)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def train(model, optimizer, x, y):
    mask = (x.sum(dim=-1) > 0).unsqueeze(dim=-1)  # 根据像素和是否为0构造掩码
    criterion = MaskedLoss(mask)

    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    return loss.item()


# 准备数据并进行训练
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)

x = torch.randn(10, 3, 32, 32)  # 10张32x32的RGB图像
y = torch.randn(10, 1, 32, 32)  # 与x对应的深度图

for i in range(100):
    loss = train(model, optimizer, x, y)
    print("epoch %d: loss = %.4f" % (i + 1, loss))