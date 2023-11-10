import cv2
import math
import urllib.request

# Load the pre-trained models
def load_models():
    # Age estimation model
    age_model_url = "https://github.com/GilLevi/AgeGenderEstimation/raw/master/models/age_net.caffemodel"
    age_model_path = "age_net.caffemodel"
    urllib.request.urlretrieve(age_model_url, age_model_path)

    # Gender classification model
    gender_model_url = "https://github.com/GilLevi/AgeGenderEstimation/raw/master/models/gender_net.caffemodel"
    gender_model_path = "gender_net.caffemodel"
    urllib.request.urlretrieve(gender_model_url, gender_model_path)

    # Face detection model
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    return age_model_path, gender_model_path, face_cascade

# Perform face detection, age, and gender estimation
def detect_age_gender(frame, face_cascade, age_net, gender_net):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face for age estimation
        age_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Perform age estimation
        age_net.setInput(age_blob)
        age_preds = age_net.forward()
        age = int(math.floor(age_preds[0][0] * 100))

        # Preprocess the face for gender classification
        gender_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Perform gender classification
        gender_net.setInput(gender_blob)
        gender_preds = gender_net.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

        # Draw rectangles and text labels on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f"Age: {age}, Gender: {gender}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Main program
def main():
    # Open the video capture
    cap = cv2.VideoCapture(0)
    
    # Load the models
    age_model_path, gender_model_path, face_cascade = load_models()

    # Load the pre-trained models
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', age_model_path)
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.protot')
