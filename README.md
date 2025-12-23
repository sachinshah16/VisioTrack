
# VisioTrack – AI-Based Real-Time Visual Assistance System

## 📌 Overview
**VisioTrack** is an AI-powered real-time object detection and distance estimation system designed to assist visually impaired individuals. The system uses computer vision and deep learning to detect nearby objects, estimate their distance using monocular vision, and provide real-time audio feedback to enhance user awareness and safety.

---

## 🎯 Problem Statement
Visually impaired users face difficulty identifying surrounding objects and judging their distance, especially in dynamic environments. Traditional assistive tools lack real-time object recognition and contextual awareness.

---

## 💡 Solution
VisioTrack solves this problem by:
- Detecting objects in real time using a deep learning model
- Estimating object distance using camera focal length
- Announcing nearby objects through text-to-speech
- Streaming live annotated video via a web interface

---

## 🧠 Key Features
- Real-time object detection using YOLOv5
- Distance estimation using monocular vision
- Audio feedback with cooldown to avoid repetition
- Live video streaming via Flask
- GPU (CUDA) support for faster inference
- Performance optimization using frame skipping
- Multithreaded speech processing

---

## 🛠️ Technologies Used

### Programming & Frameworks
- Python  
- Flask  
- OpenCV  

### AI & Deep Learning
- YOLOv5 (Ultralytics)  
- PyTorch  

### Audio
- pyttsx3 (Offline Text-to-Speech)

---

## 🏗️ System Architecture
1. Camera captures live video frames  
2. Frames are processed by YOLOv5 for object detection  
3. Object distance is estimated using focal length formula  
4. Nearest objects are prioritized  
5. Audio feedback announces object name and distance  
6. Annotated video is streamed via Flask web app  

---

## 📐 Distance Estimation Formula
Distance is calculated using:

Distance = (Known Object Width × Focal Length) / Object Width in Pixels

---

## 📂 Project Structure
VisioTrack/
├── app.py
├── model.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md

---

## 🚀 How to Run the Project

1. Clone the repository  
2. Install dependencies using `pip install -r requirements.txt`  
3. Run the application using `python app.py`  
4. Open `http://localhost:5000` in your browser  

---

## ⚙️ Hardware Requirements
- Webcam
- Optional NVIDIA GPU for CUDA acceleration

---

## 📈 Performance Optimizations
- Frame skipping to improve FPS
- Lightweight YOLOv5 Nano model
- Multithreaded text-to-speech processing
- Automatic CPU/GPU selection

---

## 🧪 Supported Objects
- Person
- Chair
- Bottle
- Cell Phone
- Mouse

---

## 🔮 Future Enhancements
- Custom-trained YOLO model
- Mobile deployment
- Depth camera integration
- Directional obstacle guidance
- Multilingual voice support

---

## 👨‍💻 Author
Sachin Kumar Shah  
B.Tech CSE (AI & ML)

---

## 📜 License
This project is intended for educational and research purposes.
