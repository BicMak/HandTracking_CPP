# Gesture-based Mouse Control

![KakaoTalk_20250831_221927089](https://github.com/user-attachments/assets/bec8ece9-cc44-427a-bf27-9ab84e738aec)

[Demo Video](https://www.youtube.com/watch?v=1VTc5y_urDM)


## Overview
This project enables mouse control through hand gestures by processing Mediapipe and Hagrid models in parallel.

The system captures hand movements via video and translates them into corresponding mouse actions.

**Key Components:**
- **Mediapipe**: Detects finger joints and skeletal structure to track precise finger positions
- **Hagrid (YOLOv10n)**: Recognizes various hand poses and gestures
- **Integration**: Visualizes hand movements using Mediapipe's coordinate data while determining mouse actions through YOLO-detected poses


##  Performance Analysis Results (37 measurements)

### Average Processing Times by Component:
- **Camera**: 6.16ms (Range: 5-8ms)
- **Reasoning**: 39.19ms (Range: 25-51ms)  
- **Visualization**: 9.35ms (Range: 0-29ms)

**Total Average Processing Time**: 54.70ms

### Performance Distribution:
- **Camera**: 11.3% of total time
- **Reasoning**: 71.6% of total time
- **Visualization**: 17.1% of total time

### Key Insights:
The reasoning component dominates the processing pipeline, accounting for over 70% of the total execution time. Camera capture is highly consistent and efficient, while visualization time varies significantly (0-29ms) depending on whether rendering is needed.



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
