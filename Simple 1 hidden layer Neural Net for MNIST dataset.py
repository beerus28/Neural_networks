
# coding: utf-8

# # Simple Neural Network trained on MNIST dataset

# In[1]:


# importing all the required libraries, ensure that there are no errors that pop up
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F


# ### Loading up MNIST dataset
# 
# MNIST handwritten digits dataset contains 60000 training images and 10000 test images. We'll be using the training images to train our model and then calculate the test accuracy. Load up train and test MNIST dataset variables from torchvision.datasets.MNIST. 
# 
# Remember to set the transform parameter to transforms.ToTensor(), we are working with Pytorch Tensors and not PIL images. 
# set the root as '../data/', i.e outside assignment directory.
# 

# In[2]:


mnist_train = None
mnist_test = None


# In[3]:


mnist_trainset = datasets.MNIST(root='../data/', train=True, download=True, transform= transforms.ToTensor())


# - root parameter = data saving location. 
# - train parameter = true as we are initializing the MNIST training dataset.
# - The download parameter = true, download it if not present in location.
# - contains raw and processed folders of data.
# 

# In[4]:


mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())


# - MNIST data-set consists of hand written digits of size 28*28px.
# - torchvision.datasets is used to download and import the data-set while 
# - torch.utils.data.DataLoader returns an iterator over the dataset.
# - Input layer consists of 28*28(784) units. 
# - Output will be of 10 units(since we predicting numbers from 0–9). 
# 

# ### Building the network model
# 
# Fill up the class SimpleNeuralNet.

# In[5]:


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(SimpleNeuralNet, self).__init__()
        self.hidden = torch.nn.Linear(input_nodes, hidden_nodes)   # hidden layer
        self.out = torch.nn.Linear(hidden_nodes, output_nodes)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        
        return x
    
    


# ### Set up some training parameters
# 

# In[6]:


batch_size = 100
input_dimension = 784
hidden_dimension = 100
output_dimension = 10
learning_rate = 0.001
num_epochs = 4


# ### Training objects
# 
# Set up the training and test dataloaders, create an instance of the neural net and identify and set the loss criterion. Use Adam optimizer to optimize the network. Also the Network we are training will have 784 input nodes, 100 hidden nodes and 10 output nodes.

# In[7]:


train_dataloader = None
test_dataloader = None
neural_net = None
criterion = None
optimizer = None


# In[8]:


# prepare data for input:

train_dataloader = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=100, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=mnist_testset,  batch_size=100, shuffle=False)


# In[9]:


neural_net = SimpleNeuralNet(input_dimension, hidden_dimension, output_dimension)     # define the network
print(neural_net)  # net architecture


# In[10]:


# Loss and Optimizer

optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()


# Now comes the training part.
# 
# - Feed Forward the network.
# - Compute loss .
# - Back-propagate to compute the gradients.
# - Update the weights with the help of optimizer.
# - Repeat(1–4) till the model converges.
# 
# 

# ### Training the model
# 
# Now that we have all the training objects in place, lets train the model for epochs defined by num_epochs. Remember to reshape the images variable before passing into the network. Also answer the following question.
# 
# #### What does the first dimension in the images and labels variable represent?
# Ans: the size of the batch. 

# In[11]:


#from torch.autograd import Variable


total_step = len(train_dataloader)

for epoch in range(num_epochs):
    #print("lol")
    for i,(images, labels) in enumerate(train_dataloader):
        
        #print(i)
        
        images = images.reshape(-1,28*28)

        out = neural_net(images)                 # input x and predict based on x
        loss = loss_func(out, labels)  

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        

        
print("finish train")


# ### Testing the model accuracy
# 
# Now that the model is trained, test the model accuracy. Print out the model accuracy, as an additional check ensure that accuracy is atleast 95% for the given parameters.

# In[12]:


with torch.no_grad():   # important precaution:
    correct = 0
    total = 0
    
    for images,labels in test_dataloader:
        images = images.reshape(-1,28*28)
        out = neural_net(images)
        _,predicted = torch.max(out.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))


# REFERENCES:
#     
# - https://medium.com/@athul929/hand-written-digit-classifier-in-pytorch-42a53e92b63e
#     
#   
