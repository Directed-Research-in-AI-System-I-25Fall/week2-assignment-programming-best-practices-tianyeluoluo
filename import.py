import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize(224),                        
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),                         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)


model = resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  
model.eval()  

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the MNIST test images: {accuracy:.2f}%')