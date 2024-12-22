import warnings

warnings.filterwarnings('ignore')

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import matplotlib.pyplot as plt
import cv2

# Load the MiDaS model and move to MPS (GPU)
midas = torch.hub.load('intel-isl/MidaS', 'DPT_Large')
midas.to('mps')
midas.eval()

# Load the necessary transforms for MiDaS
transforms = torch.hub.load('intel-isl/MidaS', 'transforms')
transform = transforms.dpt_transform

# Open the webcam feed
cam = cv2.VideoCapture(1)

while cam.isOpened():
    ret, frame = cam.read()

    if not ret:
        break

    # Flip the image and apply transformations
    img = cv2.flip(frame, 1)

    # Transform the image and move to GPU (MPS)
    img_batch = transform(img).to('mps')

    # Perform inference with MiDaS on GPU
    with torch.no_grad():
        prediction = midas(img_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        # Move prediction back to CPU for further processing and visualization
        output = prediction.cpu().numpy()

    # Display the depth map
    plt.axis('off')
    plt.imshow(output, cmap='magma')  # Visualize the depth map in grayscale
    cv2.imshow('img', frame)  # Show the original frame from the webcam
    plt.pause(0.00001)

    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()

plt.show()
