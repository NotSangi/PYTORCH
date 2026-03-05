import torch
import torch.nn as nn
import torch.optim as optim
from cnn import SimpleCNN
from data_pipeline import train_loader, val_loader, test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f'Using: {device}')

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def train_model(model, loader):
    model.train()
    running_loss = 0
    total = 0
    correct = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

def evaluate_model(model, loader):
    model.eval()
    running_loss = 0
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy


epochs = 25
for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_loader)
    val_loss, val_acc = evaluate_model(model, val_loader)
    
    print(f'Epoch [{epoch+1}/{epochs}]')
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%')
    print('-'*40)
    
    