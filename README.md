# **Sign Language Detector**  

## **Overview**  
This repository contains a **Sign Language Detector** that utilizes **MediaPipe Hands** for real-time hand tracking and a **Random Forest Classifier** for recognizing hand gestures. The system captures gestures using a webcam, processes hand landmarks, and predicts the corresponding sign.  

## **Features**  
- **Real-Time Gesture Recognition**: Uses a trained **Random Forest Classifier**.  
- **Hand Landmark Extraction**: Uses **MediaPipe Hands** to track keypoints.  
- **Dataset Collection**: Captures and stores images for training.  
- **Model Training**: Trains a classification model on collected hand landmarks.  

## **Datasets Used**  
The system requires images of different hand gestures stored in the `data/` directory. The dataset consists of:  
- **8 Gesture Classes**: Good Job, OK, Good Luck, Dislike, Peace, Hello, Excuse Me, I Love You.  
- **100 Images per Class**: Each class has 100 images captured from a webcam.  

## **Installation & Setup**  

### **Prerequisites**  
Follow these steps to set up and run the system:  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/PadamArora/SignLanguageDetector.git
   cd SignLanguageDetector
   ```  

2. **Install dependencies**  
   ```bash
   pip install opencv-python mediapipe scikit-learn numpy matplotlib
   ```  

## **How It Works**  

### **1. Data Collection**  
Run the following script to **capture images** for different gesture classes:  
```bash
python collect_images.py
```
- Prompts the user to **press 'q'** to start capturing images.  
- Saves **100 images per class** inside `data/` (e.g., `data/0/`, `data/1/`, …).  

### **2. Create Dataset**  
Once images are collected, extract **hand landmarks** using MediaPipe:  
```bash
python create_dataset.py
```
- Reads images from `data/` and extracts **21 hand keypoints**.  
- Saves processed data in **`data.pickle`**.  

### **3. Train the Model**  
Train a **Random Forest Classifier** on the dataset:  
```bash
python train_classifier.py
```
- Splits data into **80% training & 20% testing**.  
- Saves the trained model as **`model.p`**.  

### **4. Real-Time Gesture Recognition**  
Once the model is trained, run the script to **detect gestures via webcam**:  
```bash
python inference_classifier.py
```
- Detects hand gestures in real-time.  
- **Draws bounding boxes & labels** around detected hands.  
- Press **'q'** to exit.  

## **Example Output**  
```bash
Predicted Gesture: Peace ✌️  
Predicted Gesture: I Love You 🤟  
```  

## **Future Enhancements**  
- Improve accuracy using **Deep Learning (CNNs)**.  
- Add support for **two-hand gestures**.  
- Expand dataset with more gestures.  

## **Contributing**  
Contributions are welcome! Feel free to **fork the repository** and submit a **pull request**.  

## **License**  
This project is licensed under the **MIT License**.
