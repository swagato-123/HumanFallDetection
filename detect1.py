import matplotlib.pyplot as plt
import torch
import cv2
import math
from torchvision import transforms
import numpy as np
import os


from tqdm import tqdm

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.general import strip_optimizer

from utils.plots import output_to_keypoint, plot_skeleton_kpts

import argparse

def fall_detection(poses):

    for pose in poses:

        xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
        xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)

        left_shoulder_y = pose[23]
        left_shoulder_x = pose[22]
        right_shoulder_y = pose[26]

        left_hip_y = pose[41]
        left_hip_x = pose[40]
        right_hip_y = pose[44]

        len_factor = math.sqrt(((left_shoulder_y - left_hip_y) ** 2 + (left_shoulder_x - left_hip_x) ** 2))

        left_ankle_y = pose[53]
        right_ankle_y = pose[56]

        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx

        if (left_shoulder_y > left_ankle_y - len_factor) and left_hip_y > left_ankle_y - (len_factor / 2) and left_shoulder_y > left_hip_y - (len_factor / 2) or (
                right_shoulder_y > right_ankle_y - len_factor and right_hip_y > right_ankle_y - (
                len_factor / 2) and right_shoulder_y > right_hip_y - (len_factor / 2)) \
                or difference < 0:
            
            return True, (xmin, ymin, xmax, ymax)

    return False, None

# Given the image (Containing the fall), BB of the human()
# Draws a rectangular BB around that person
def falling_alarm(image, bbox):

    x_min, y_min, x_max, y_max = bbox
    
    # Draw a BB around the given frame
    cv2.rectangle(
        img=image, pt1=(int(x_min), int(y_min)), pt2=(int(x_max), int(y_max)), 
        color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA
    )
    
    # Display the warning near the top left corner (Slightly below)
    cv2.putText(
        img=image, text='Person Fell down', 
        org=(11, 100), fontFace=0, fontScale=1, color=[0, 0, 2550], 
        thickness=3, lineType=cv2.LINE_AA
    )


# For instantiating the YoloV7 Pose estimation model, device wherein the model should be loaded (GPU or CPU)
# Returning the model, device
def get_pose_model(device):

    # We shall load the model on the GPU(If available), 
    # Else revert to the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # Load the model weights
    # weights = Dictionary containing the weights of the pre-trained YOLOv7 model.
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    
    # Load the YoloV7 Pose model
    # Extract the model architecture from the weights dictionary
    model = weigths['model']

    # Set the model in evaluation mode
    # float() method called upon the model instance, to ensure that the weights are stored in 32-bit floating-point format.
    _ = model.float().eval()

    # If GPU is available, The model is loaded onto the GPU
    # Converted to half-precision floating-point format for faster computations.
    if torch.cuda.is_available():
        model = model.half().to(device)

    # Return the model, device where the model is loaded
    return model, device

# Drawing the Pose landmarks
def get_pose(image, model, device):
    #print(f"Image dimensions before preprocessing = {image.shape}")
    image = letterbox(image, 960, stride=64, auto=True)[0]
    #print(f"Image dimensions after preprocessing = {image.shape}")

    image = transforms.ToTensor()(image)

    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)

    with torch.no_grad():
        output, _ = model(image)

    output = non_max_suppression_kpt(
        output, conf_thres=0.25, iou_thres=0.65, 
        nc=model.yaml['nc'], 
        nkpt=model.yaml['nkpt'],
        kpt_label=True
    )

    with torch.no_grad():
        output = output_to_keypoint(output)

    return image, output


"""
Preparing the image for display

Takes in an image tensor as input
Converts it into a format supported by OpenCV for display
"""
def prepare_image(image):
    """
    Change from format [b, c, h, w] --> [h, w, c]

    image[0] = Selects the first image in the batch
    permute(1, 2, 0) :- Rearranges the dimensions of the Tensor from [c, h, w] TO [h, w, c]

    .permute(1, 2, 0) * 255 :- 
        Multiplies the tensor by 255 to scale the normalized tensor values in [0, 1] to [0, 255]        
    """
    _image = image[0].permute(1, 2, 0) * 255

    # Move the image from GPU to CPU memory
    # Then convert it to a numpy array
    _image = _image.cpu().numpy().astype(np.uint8)

    # Convert the color format of the numpy array from RGB colorspace to BGR
    # Cuz this is the format expected by OpenCV for displaying images
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)

    # Return the frame(image) for
    return _image


"""
Prepare the video writer object that will write the output video 
with pose keypoints and fall detection alarms.
"""
def prepare_vid_out(video_path, vid_cap):

    vid_write_image = letterbox(
        vid_cap.read()[1], 
        960, stride=64, auto=True
    )[0]

    resize_height, resize_width = vid_write_image.shape[:2]

    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}_keypoint.mp4"

    """
    filename=Input video file
    fourcc = Specifying the codec of the .mp4 video file
    fps = Frame rate of the video stream
    """
    out = cv2.VideoWriter(
        filename=out_video_name, 
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'), 
        fps=30, 
        frameSize=(resize_width, resize_height)
    )

    return out

"""
Video acquisition and processing (Videos)
Not intended for display
"""

"""
Video acquisition and processing (Webcam)
Not intended for display
"""
def processVideo(source, device):
    pass

def processWebcam(source, device):

    # Load the YoloV7 Pose model, device on which the model is to be loaded
    model, device = get_pose_model(device)

    """
    filename=Input video file
    fourcc = Specifying the codec of the .mp4 video file
    fps = Frame rate of the video stream
    """
    out_video_name = "web_cam_keypoint.mp4"


    vid_out = cv2.VideoWriter(
        filename=out_video_name, 
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'), 
        fps=30, 
        frameSize=(960, 960)
    )

    # _frames = List storing each of the video frames
    _frames = []
    
    while True:
        success, frame = source.read()
        _frames.append(frame)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    source.release()
    cv2.destroyAllWindows()

    """
    Now perform Pose estimation, Fall detection on each of the frames in the _frames list

    tqdm :- 
        For displaying the progress bar (Visual indicator of the progress made thus far)
        We can see a progress bar that updates in real-time as the frames are being processed.
    """

    for image in tqdm(_frames):
        image, output = get_pose(image, model, device)
        _image = prepare_image(image)
        is_fall, bbox = fall_detection(output)
        
        if is_fall:
            falling_alarm(_image, bbox)

        vid_out.write(_image)

    vid_out.release()


def process_video(source, device):

    # Initialize the video capture object
    if source.isnumeric():    
        vid_cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
        processWebcam(vid_cap, device)
    else:
        vid_cap = cv2.VideoCapture(source)

    # Check if the video_cap object is properly initialized (Check if the video file is properly opened)
    if not vid_cap.isOpened():
        print('Error while trying to read video. Alter or check the path provided')
        return

    # Load the YoloV7 Pose model, device on which the model is to be loaded
    model, device = get_pose_model(device)
    vid_out = prepare_vid_out(source, vid_cap)

    # Read the 1st frame
    success, frame = vid_cap.read()

    # _frames = List storing each of the video frames
    _frames = []

    # Store each frame in _frames list
    while success:
        _frames.append(frame)

        # Read successive frames from the video
        success, frame = vid_cap.read()

    """
    Now perform Pose estimation, Fall detection on each of the frames in the _frames list

    tqdm :- 
        For displaying the progress bar (Visual indicator of the progress made thus far)
        We can see a progress bar that updates in real-time as the frames are being processed.
    """

    for image in tqdm(_frames):
        image, output = get_pose(image, model, device)
        _image = prepare_image(image)
        is_fall, bbox = fall_detection(output)
        
        if is_fall:
            falling_alarm(_image, bbox)

        vid_out.write(_image)

    vid_out.release()
    vid_cap.release()


def from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./fall_dataset/videos/video_7.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = from_command_line()

    # Unpack values from the opt object
    source = opt.source
    device = opt.device

    # 
    process_video(source, device)