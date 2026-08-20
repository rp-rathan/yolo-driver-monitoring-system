# Real-Time Driver Monitoring System

A real-time **Driver Monitoring System (DMS)** built using **Python, OpenCV, and Ultralytics YOLO**.

The system uses a webcam to monitor the driver and detect potentially unsafe driving behaviors such as:

- Drowsiness
- Mobile phone usage
- Seat-belt status
- Awake/alert state

The system provides real-time visual feedback through bounding boxes and status banners.

---

## Features

- Real-time webcam processing
- YOLO object detection and tracking
- Custom YOLO model support
- USB webcam support
- Automatic camera fallback
- Drowsiness detection
- Mobile phone detection
- Seat-belt detection
- Awake-state detection
- Real-time warning system
- Temporal detection smoothing
- Persistent driver bounding box
- Confidence scores
- Object tracking IDs
- Real-time driver status banner

---

## System Architecture

```text
                    USB Camera
                        |
                        v
               OpenCV Video Capture
                        |
                        v
                 Video Frame
                   Processing
                        |
                        v
                YOLO Detection
                        |
                        v
                 YOLO Tracking
                        |
                        v
              Driver State Analysis
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
          Awake       Drowsy      Mobile
            |           |           |
            v           v           v
          GREEN         RED         RED
                        |
                        v
               Driver Monitoring UI
