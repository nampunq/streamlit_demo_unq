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
model.load_state_dict(torch.load('./image_classifier.pth'))
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

def intro():
    st.title('Image classifier')
    st.write('This is a website used to deploy a model using Streamlit. We will import a picture and make it predict what it is.')

# HEADER 1: General infromation of the model
def header1():
    st.header("Architecture of the model")
    tab1, tab2 = st.tabs(["📋 Image", "🖊️ Code"])
    with tab1:
        st.image('./flow.png')
    with tab2:
        code_text = """
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        """
        st.code(code_text)

    container = st.container(border=True)
    container.write("""The model may not be accurate in terms of architecture, but this is mostly how CNN works.
                    It goes through multiple convolutional and pooling layers. Once it finishes extracting important information, 
                    it gives a probability for each class, the highest one being the prediction of the image""")

    st.write('You can modify the model based on your understanding and make it better :)')

# HEADER 2: General information of the dataset
def header2():
    st.header("Dataset")
    st.markdown("**Name**: CIFAR100")
    st.markdown("**Number of data**: 60000 images")
    st.markdown("**Classes**: 10, each with 6000 images/class")

# HEADER 3: Showing more utils of Streamlit
def header3():
    st.header("Train your own model")
    st.caption("""This function is under development""")

    with st.expander("if you want to see how it looks, click here!"):

        number = st.number_input('Epochs', min_value=0, max_value=None, value=0, step=1)
        st.write("The number of epochs is ", number)

        if st.button('Train Model', disabled=(number < 1)):
            progress_text = "Operation in progress. Please wait."
            progress_bar = st.progress(0, text=progress_text)
            
            for percent in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent + 1, text=progress_text)
            progress_done = "Finish training with your own epochs"    
            progress_bar.progress(value=100, text=progress_done)

# HEADER 4: Testing the model with the built-in import function and return the result
def header4():
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
            
intro()
header1()
header2()
header3()
header4()