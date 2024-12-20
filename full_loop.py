#!/usr/bin/env python
# coding: utf-8

# # Load Vision Model
# 
# First we load in the data and our model into memory

# **Important:** For this notebook to function it needs to be execuetd with `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/$LD_PRELOAD` else you will get a error. 
# I included this in jupyter by adding a docker environment variable: `-e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`.

# In[1]:


import numpy as np


# In[2]:


import torch

from utils.dataloader import VOCDataLoader
loader = VOCDataLoader(train=False, batch_size=1)


# In[3]:


from tinyyolov2 import TinyYoloV2
from utils.yolo import nms, filter_boxes
from utils.viz import display_result

# make an instance with 20 classes as output
net = TinyYoloV2(num_classes=20)

# load pretrained weights
sd = torch.load("voc_pretrained.pt")
net.load_state_dict(sd)

#put network in evaluation mode
net.eval()


# # Define Camera Callback

# In[4]:


from utils.camera import CameraDisplay
import time
import cv2
now = time.time()


# In[5]:


def get_predictions(image):
    # Ensure the input tensor is correct
    if image.shape != (1, 3, 320, 320):
        raise ValueError(f"Invalid input shape: {image.shape}, expected (1, 3, 320, 320)")

    # Pass through the network
    with torch.no_grad():
        output = net(image)

    # Process YOLO output
    output = filter_boxes(output, 0.001)
    output = nms(output, 0.1)

    # Convert the tensor back to a NumPy array for OpenCV
    image_np = image.squeeze(0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)  # Scale back to 0-255 and convert to uint

    # Visualize the results if there are any detections
    img_shape = 320 
    if output:
        bboxes = torch.stack(output, dim=0)
        for i in range(bboxes.shape[1]):
            if bboxes[0, i, -1] >= 0:
                # Debug
                print(f"Detection detected: class {num_to_class(int(bboxes[0, i, 5]))}, confidence {bboxes[0, i, 4]:.2f}")
                
                # Calculate bounding box coordinates
                cx = int(bboxes[0, i, 0] * img_shape - bboxes[0, i, 2] * img_shape / 2)
                cy = int(bboxes[0, i, 1] * img_shape - bboxes[0, i, 3] * img_shape / 2)
                w = int(bboxes[0, i, 2] * img_shape)
                h = int(bboxes[0, i, 3] * img_shape)
    
                # Draw rectangle on the image
                cv2.rectangle(
                    image,  # OpenCV image (numpy array)
                    (cx, cy),  # Top-left corner
                    (cx + w, cy + h),  # Bottom-right corner
                    (0, 0, 255),  # Color (BGR) - Red in this case
                    2  # Thickness
                )
    
                # Add label text
                label = f"{num_to_class(int(bboxes[0, i, 5]))} {bboxes[0, i, 4]:.2f}"
                cv2.putText(
                    image,
                    label,
                    (cx, cy - 10),  # Slightly above the rectangle
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    0.5,  # Font scale
                    (0, 0, 255),  # Color (BGR) - Red
                    1  # Thickness
                )
                
    return image_np


# In[6]:


# Define a callback function (your detection pipeline)
# Make sure to first load all your pipeline code and only at the end init the camera

def callback(image):
    global now

    if image is None:
        raise ValueError("Received empty frame from the camera.")

    # Resize and preprocess the image
    img_resized = cv2.resize(image, (320, 320))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Predict and visualize
    img_with_predictions = get_predictions(img_tensor)

    # Ensure the image is a NumPy array for OpenCV
    img_with_predictions = img_with_predictions.numpy() if isinstance(img_with_predictions, torch.Tensor) else img_with_predictions

    # Add FPS to the image
    fps = f"{int(1 / (time.time() - now))}"
    now = time.time()
    cv2.putText(img_with_predictions, f"fps={fps}", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    return img_with_predictions


# In[7]:


# Initialize the camera with the callback
cam = CameraDisplay(callback)


# # Camera Loop

# In[8]:


# The camera stream can be started with cam.start()
# The callback gets asynchronously called (can be stopped with cam.stop())
cam.start()


# Execute below, to stop camera loop.
# 
# ----------------------------------------------------------------

# In[7]:


# The camera should always be stopped and released for a new camera is instantiated (calling CameraDisplay(callback) again)
cam.stop()
cam.release()


# In[ ]:




