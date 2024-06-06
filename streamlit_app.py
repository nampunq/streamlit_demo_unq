import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch.nn as nn
import torch.nn.functional as F
import time

# Neural network definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load pre-trained model
model = Net()
model.load_state_dict(torch.load('image_classifier.pth'))
model.eval()

# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#TODO

st.title('Image classifier')
st.write('This is a website used to deploy a model using Streamlit. We will import a picture and make it predict what it is.')
st.write('Here\'s the model defined in the code:')

code_text = """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
"""
st.code(code_text)
st.write('We want to use a simple model that can run on multiple epochs. We will have a textbox to allow users to choose the number of epochs')

number = st.number_input('Epochs', min_value=None, max_value=None, value=5, step=1)
st.write("The number of epochs is ", number)

if st.button('Train Model'):
    progress_text = "Operation in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    
    for percent in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent + 1, text=progress_text)
    progress_done = "Finish training your own epochs"    
    progress_bar.progress(value=100, text=progress_done)

file_upload = st.file_uploader("Choose file", 
                               type = ['png', 'jpg', 'jpeg'])

if file_upload is not None:
    image = Image.open(file_upload).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    input_image = transform(image)
    input_batch = input_image.unsqueeze(0)
    
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    
    _, predicted = torch.max(output, 1)
    with st.spinner(text='In progress'):
        time.sleep(3)
        st.success('Done')
        st.write(f"Prediction: {classes[predicted]}")