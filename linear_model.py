import torch
import torch.nn as nn
import torch.optim as optim

distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1,1))

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(distances)
    loss = loss_function(outputs, times)
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    new_x = torch.tensor([[2.0]], dtype=torch.float32)
    prediction = model(new_x)
    print(prediction.item())
    