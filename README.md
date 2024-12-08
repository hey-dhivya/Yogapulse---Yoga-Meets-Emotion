##**YogaPulse - Yoga Meets Emotion 🌸**
Overview
YogaPulse is an AI-powered yoga application built using Streamlit, PyTorch, and Mediapipe. It combines pose detection, proficiency scoring, and sentiment analysis to deliver a personalized yoga and wellness experience. Users can upload their yoga pose images, analyze their proficiency, and receive recommendations based on their detected emotions.

Features
Pose Detection and Classification

Analyzes uploaded images to detect and classify yoga poses.
Predicts yoga poses using a fine-tuned ResNet-18 model.
Provides a proficiency score based on pose landmarks' visibility.
Mood Analysis

Uses a HuggingFace sentiment analysis pipeline to assess user emotions based on their input.
Suggests personalized yoga routines tailored to the detected mood (positive, neutral, or negative).
Wellness Insights

Tracks session progress, including analyzed poses and proficiency scores.
Generates a downloadable personalized wellness report.
User Interface

Easy-to-use, visually appealing interface with styled components.
Real-time feedback and pose analysis.
Installation and Setup
Prerequisites
Python 3.8 or later
torch, torchvision, and other required libraries
Pretrained model weights (model.pth)
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/your-repo/yogapulse.git
cd yogapulse
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Place the model.pth file in the project directory.

Running the Application
Start the Streamlit server by running:

bash
Copy code
streamlit run app.py
The application will be accessible at http://localhost:8501 in your browser.

How to Use
Upload Image: Click on the "Choose an image" button and upload a yoga pose image.
Pose Analysis: View the detected pose, proficiency score, and annotated image.
Mood-Based Suggestions:
Enter your current mood in the text box.
Receive yoga routine recommendations tailored to your mood.
Download Wellness Report: Click the "Generate Personalized Report" button to download session insights.
