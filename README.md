# Gesture-based Mouse Control


https://www.youtube.com/watch?v=1VTc5y_urDM

## Overview
This project enables mouse control through hand gestures by processing Mediapipe and Hagrid models in parallel.
The system captures hand movements via video and translates them into corresponding mouse actions.

**Key Components:**
- **Mediapipe**: Detects finger joints and skeletal structure to track precise finger positions
- **Hagrid (YOLOv10n)**: Recognizes various hand poses and gestures
- **Integration**: Visualizes hand movements using Mediapipe's coordinate data while determining mouse actions through YOLO-detected poses

The system operates at an average frame rate of 12 Hz, providing real-time gesture-to-mouse translation.

## System Architecture Flow:

1. **Input Stage**
   - Camera frame acquisition captures video frames

2. **Primary Processing Stage** 
   - Acquired frames branch into three directions:
     - MediaPipe processing (hand landmark detection)
     - YOLO processing (object detection)
     - Update visualize model (visualization model update)

3. **Parallel Processing Stage**
   - MediaPipe results branch into 2 parallel tasks:
     - Control the mouse (mouse control)
     - Visualize the boundbox (bounding box visualization)

4. **Integration and Output Stage**
   - All parallel processing results are finally integrated into "update image"
   - Processed results are displayed on screen

**Key Features:**
- **Parallel Processing**: MediaPipe and YOLO execute simultaneously for performance optimization
- **Multi-tasking**: Single MediaPipe result enables simultaneous mouse control and visualization
- **Real-time Processing**: All results are integrated into one image providing real-time feedback


## Main Function
- Point up: move the mouse cursor
- Fist: click the left mouse button
- Captures hand landmarks and gestures and visualizes the bounding box and finger joint points
- Tracks fingertip positions

## References


### Models & Frameworks
- [Mediapipe](https://huggingface.co/STMicroelectronics/hand_landmarks) - Hand landmark detection
- [Hagrid YOLOv10n](https://github.com/hukenovs/hagrid) - Object detection backbone
