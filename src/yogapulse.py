import time
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from transformers import pipeline  # HuggingFace sentiment analysis pipeline
from io import BytesIO
import pandas as pd  # Optional: To create a data frame for tabular reports

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for your specific number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # Assuming your model was trained on 5 classes

# Load your trained model weights
model.load_state_dict(torch.load('model.pth', map_location=device))  # Load your trained model weights
model = model.to(device)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 (standard for ResNet)
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert image to RGB if not already
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ResNet standard
])

# Define custom class names (modify these to match your model's classes)
class_names = ["warrior", "tree", "plank", "goddess", "downdog"]

# Initialize session state
if "session_start_time" not in st.session_state:
    st.session_state["session_start_time"] = time.time()
if "pose_analysis" not in st.session_state:
    st.session_state["pose_analysis"] = []
if "mood_label" not in st.session_state:
    st.session_state["mood_label"] = None
if "mood_score" not in st.session_state:
    st.session_state["mood_score"] = None

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit interface
st.set_page_config(page_title="YogaPulse - Yoga Meets Emotion", page_icon="🧘", layout="wide")
st.markdown("""
    <style>
        .css-1d391kg { 
            font-family: 'Arial', sans-serif;
            color: #3b3b3b;
            text-align: center;
        }
        .css-ffhzg2 { 
            font-size: 30px;
            color: #FF6F61;
        }
        .css-1d391kg p {
            font-size: 18px;
            color: #4A4A4A;
        }
        .stButton button {
            background-color: #FF6F61;
            color: white;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #E55C4E;
        }
        .stFileUploader {
            padding: 20px;
            background-color: #F3F3F3;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("YogaPulse - Yoga Meets Emotion 🌸")
st.write("Upload an image for your pose correction and get personalized recommendations for your emotional state!")

# --- Image Upload and Pose Analysis ---
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    transformed_img = transform(img)
    transformed_img = transformed_img.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(transformed_img)
        _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    proficiency_score = None
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_cv, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            visibility_scores = [lm.visibility for lm in results.pose_landmarks.landmark]
            proficiency_score = np.mean(visibility_scores) * 100

    image_with_pose = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    st.session_state["pose_analysis"].append({
        "pose": predicted_class,
        "proficiency_score": proficiency_score,
        "timestamp": time.time()
    })

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(image_with_pose, caption="Pose Detection", use_column_width=True)

    st.write(f"**Predicted Class:** {predicted_class}")
    if proficiency_score is not None:
        st.write(f"**Pose Proficiency:** {proficiency_score:.2f}%")

# --- Mood-Based Session Recommendation ---
st.subheader("Session Recommendation 💆")
user_mood = st.text_input("How are you feeling today? (e.g., happy, stressed, relaxed)")

if user_mood:
    sentiment_result = sentiment_analyzer(user_mood)
    mood_label = sentiment_result[0]["label"]
    mood_score = sentiment_result[0]["score"]

    st.session_state["mood_label"] = mood_label
    st.session_state["mood_score"] = mood_score

    st.write(f"**Detected Mood:** {mood_label} (Confidence: {mood_score:.2f})")
    st.write("**Recommended Yoga Routine:**")
    if mood_label == "POSITIVE":
        st.write("- **Happy Flow:** Try joyful poses like Warrior and Tree for energy.")
    elif mood_label == "NEGATIVE":
        st.write("- **Calm & Relax:** Try Plank and Downdog for grounding and stress relief.")
    elif mood_label == "NEUTRAL":
        st.write("- **Balanced Session:** Combine a mix of Goddess and Downdog for balance.")

# --- Personalized Wellness Insights ---
if st.button("Generate Personalized Report 📄"):
    session_duration = time.time() - st.session_state["session_start_time"]
    poses_analyzed = len(st.session_state["pose_analysis"])
    avg_proficiency = np.mean([item["proficiency_score"] for item in st.session_state["pose_analysis"]])

    mood_label = st.session_state.get("mood_label", "Not provided")
    mood_score = st.session_state.get("mood_score", 0)

    report = f"""
    ### Personalized Wellness Report
    **Session Duration:** {session_duration // 60:.0f} min {session_duration % 60:.0f} sec  
    **Poses Analyzed:** {poses_analyzed}  
    **Average Pose Proficiency:** {avg_proficiency:.2f}%  
    **Detected Mood:** {mood_label} (Confidence: {mood_score:.2f})  

    ### Recommendations
    """
    if avg_proficiency < 70:
        report += "- Focus on improving pose accuracy with guided videos.\n"
        report += "- Practice foundational poses for better stability."
    else:
        report += "- Great job! Keep practicing to maintain proficiency.\n"
        report += "- Explore advanced poses to further challenge yourself."

    # Save report to a text file
    report_file = BytesIO()
    report_file.write(report.encode('utf-8'))
    report_file.seek(0)

    st.download_button(
        label="Download Report",
        data=report_file,
        file_name="wellness_report.txt",
        mime="text/plain"
    )
