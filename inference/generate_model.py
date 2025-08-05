import torch
import torch.nn as nn
import numpy as np

# Define a simple linear regression model
class DummyRegressor(nn.Module):
    def __init__(self):
        super(DummyRegressor, self).__init__()
        self.linear = nn.Linear(1, 1)  # y = wx + b

    def forward(self, x):
        return self.linear(x)

# Instantiate model and generate dummy data
model = DummyRegressor()
X = torch.tensor(np.random.rand(100, 1) * 10, dtype=torch.float32)
y = 3.5 * X + 5 + torch.randn(100, 1) * 0.5  # True function with noise

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Save model state_dict
torch.save(model.state_dict(), "dummy_model.pt")
