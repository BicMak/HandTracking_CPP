# Gesture-based Mouse Control



## Overview
This project enables mouse control through hand gestures by processing Mediapipe and Hagrid models in parallel.
The system captures hand movements via video and translates them into corresponding mouse actions.

**Key Components:**
- **Mediapipe**: Detects finger joints and skeletal structure to track precise finger positions
- **Hagrid (YOLOv10n)**: Recognizes various hand poses and gestures
- **Integration**: Visualizes hand movements using Mediapipe's coordinate data while determining mouse actions through YOLO-detected poses

The system operates at an average frame rate of 12 Hz, providing real-time gesture-to-mouse translation.


## Main Function
- Point up: move the mouse cursor
- Fist: click the left mouse button
- Captures hand landmarks and gestures and visualizes the bounding box and finger joint points
- Tracks fingertip positions

## Configuration Method
- Check the configuration:
> 1. MpConfig: relates to the Mediapipe gesture recognition model
> 2. VideoConfig: configures the OpenCV Capture class
> 3. visualizer_config: turns the visualization function on/off

## References

### Models & Frameworks
- [Mediapipe](https://huggingface.co/STMicroelectronics/hand_landmarks) - Hand landmark detection
- [Hagrid YOLOv10n](https://github.com/hukenovs/hagrid) - Object detection backbone
