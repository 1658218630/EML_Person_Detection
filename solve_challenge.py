#!/usr/bin/env python
# coding: utf-8

# # Person-only detection

# In[3]:


import torch

from utils.dataloader import VOCDataLoaderPerson
loader = VOCDataLoaderPerson(train=False, batch_size=1)


# In[4]:


import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# Visualization to check if the data are correct loaded
for img, target in loader:
    img = F.to_pil_image(img[0])
    print("Target vector:", target[0])

    plt.imshow(img)
    plt.show()

    break  


# In[7]:


from tinyyolov2 import TinyYoloV2

# load pretrained model
model = TinyYoloV2(num_classes=20)
model.load_state_dict(torch.load('./voc_pretrained.pt'))
model.eval()


# In[8]:


import torch.nn as nn

# redefine the last layer
num_anchors = 5  
model.conv9 = nn.Conv2d(in_channels=1024, out_channels=num_anchors * (5 + 1), kernel_size=1, stride=1, padding=0)

print(model.conv9)


# In[10]:


# freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Only retrain the last layer
for param in model.conv9.parameters():
    param.requires_grad = True


model.train()



# In[ ]:




# In[11]:


# Save the retrained model
torch.save(model.state_dict(), 'tinyyolo_person_only.pt')


# In[ ]:





# In[ ]:




