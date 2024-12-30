import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.custom_model import CifarNet
from utils.data_loader import get_dataloaders
from config import *
from torch.optim.lr_scheduler import OneCycleLR

def train(model, train_loader, optimizer, criterion, scheduler, device):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            f'Loss={train_loss/(batch_idx+1):0.6f} Batch_id={batch_idx} '
            f'Accuracy={100*correct/processed:0.2f} '
            f'LR={scheduler.get_last_lr()[0]:.6f}'
        )

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return accuracy

def main():
    torch.manual_seed(42)
    
    train_loader, test_loader = get_dataloaders(
        train_transforms, 
        test_transforms, 
        BATCH_SIZE
    )
    
    model = CifarNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # One Cycle Learning Rate Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Spend 30% of time in warmup
        div_factor=10,  # Initial lr = max_lr/10
        final_div_factor=100,  # Min lr = initial_lr/100
    )
    
    criterion = nn.NLLLoss()
    
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch+1}")
        train(model, train_loader, optimizer, criterion, scheduler, DEVICE)
        accuracy = test(model, test_loader, criterion, DEVICE)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    main() 