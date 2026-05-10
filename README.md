# SmartParking AI System

The SmartParking AI System is a scalable, computer vision-based application designed to convert standard aerial CCTV feeds into actionable, real-time parking intelligence. Traditional parking facilities often rely on expensive physical sensors in every parking bay. Our project eliminates that heavy infrastructure by using artificial intelligence instead.  

This edge-based parking management system utilizes a lightweight computer vision model to analyze standard CCTV video footage in real time.  

---
<img width="600" height="338" alt="demo" src="https://github.com/user-attachments/assets/2c6b81f6-9ffe-47b1-9269-40fbba56f26f" />

<img width="2677" height="1645" alt="image" src="https://github.com/user-attachments/assets/3f382aeb-c10e-4634-8beb-086e4f749fa7" />

## 🧠 Core Architecture & My Contributions
This system was developed collaboratively. My primary responsibilities centered on the intelligence and data persistence layers:

* **Computer Vision Logic:** Led the training of the YOLO prototype and established the core computer vision logic. This involved fine-tuning the YOLOv11s deep learning model for accurate vehicle detection.  
* **Database Engineering:** Structured the SQLite database schemas. The system automatically tracks every occupancy state change and stores them locally.  
* **Technical Documentation:** Authored the AI Techniques and Implementation section, the Abstract, and structured the technical documentation.  

## ⚙️ System Features & Performance

* **Real-Time Edge Inference:** Operating within strict edge-computing constraints, the system achieved a 98.1% Mean Average Precision (mAP@50) with an inference latency of just 4.9 ms per frame.  
* **Spatial Logic Filtering:** The system relies on deterministic Region of Interest (ROI) mapping to reliably determine parking spot occupancy. This distinguishes between parked vehicles and dynamic aisle traffic, significantly reducing false occupancy readings.  
* **Temporal Optimization:** To ensure the dashboard and database remain highly responsive, the system implements a frame-skipping heuristic. State changes are only committed to the SQLite database every third frame, preventing massive pipeline bottlenecks and memory overhead.  
* **Interactive Dashboard:** The system provides a seamless, user-friendly dashboard display for users and operators to monitor lot capacity and export database logs.  

---

## 🛠️ Tech Stack

* **Deep Learning & Vision:** Python, YOLOv11s, OpenCV   
* **Data Management:** SQLite, Pandas   
* **Frontend UI:** Streamlit, Plotly   

---

## 🚀 Setup & Installation

### Prerequisites
* **Operating System:** Windows 10/11, macOS, or Linux.  
* **Python:** Python 3.9 or higher installed on your system.  

### Installation Steps

1. **Extract the Project:** Extract the provided project folder to your preferred location.  
2. **Install Dependencies:** With your virtual environment active, run the following command in your terminal:  
   ```bash
   pip install streamlit ultralytics opencv-python-headless pandas plotly
3. **Launch the Application:** In your active terminal, ensure you are in the main project directory and run:
    ```bash
    streamlit run app.py
4. Access the Dashboard: Streamlit will automatically open a new tab in your default web browser. If it does not, copy the Local URL (e.g., http://localhost:8501) from the terminal and paste it into your browser.
