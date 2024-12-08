# **YogaPulse - Yoga Meets Emotion 🌸**
## **Overview**
YogaPulse is an AI-powered yoga application built using Streamlit, PyTorch, and Mediapipe. It combines pose detection, proficiency scoring, and sentiment analysis to deliver a personalized yoga and wellness experience. Users can upload their yoga pose images, analyze their proficiency, and receive recommendations based on their detected emotions.

## **Features**
### **1.Pose Detection and Classification**
  Analyzes uploaded images to detect and classify yoga poses.
  Predicts yoga poses using a fine-tuned ResNet-18 model.
  Provides a proficiency score based on pose landmarks' visibility.

### **2.Mood Analysis**
  Uses a HuggingFace sentiment analysis pipeline to assess user emotions based on their input.
  Suggests personalized yoga routines tailored to the detected mood (positive, neutral, or negative).

### **3.Wellness Insights**
  Tracks session progress, including analyzed poses and proficiency scores.
  Generates a downloadable personalized wellness report.

## **How to Use this app**
- **1.Upload Image:** Click on the "Choose an image" button and upload a yoga pose image.
- **2.Pose Analysis:** View the detected pose, proficiency score, and annotated image.
- **3.Mood-Based Suggestions:** Enter your current mood in the text box. Receive yoga routine recommendations tailored to your mood.
- **4.Download Wellness Report:** Click the "Generate Personalized Report" button to download session insights.
