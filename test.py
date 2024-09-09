import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define a convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Define a fully connected layer
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        # Apply the convolutional layer with ReLU activation
        x = F.relu(self.conv1(x))
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Apply the fully connected layer
        x = self.fc1(x)
        return x

# Instantiate the model and move it to the device
model = SimpleCNN().to(device)
# Create a random input tensor (batch size: 1, channels: 1, height: 28, width: 28)
input_tensor = torch.randn(1, 1, 28, 28).to(device)

# Pass the input tensor through the model
output = model(input_tensor)

# Print the output
print(output)
