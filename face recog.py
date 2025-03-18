import streamlit as st
import numpy as np
import face_recognition
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from io import BytesIO

# Initialize or Load Existing Data
data_file = "registered_faces.xlsx"
try:
    df = pd.read_excel(data_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name", "Registration Number", "Mobile Number", "Year", "Branch", "Section", "Encoding"])

def capture_face_encodings(image):
    if image is not None:
        image_bytes = BytesIO(image.read())
        frame = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        if face_encodings:
            return face_encodings[0]
    return None

def plot_registration_status(success):
    fig = make_subplots(rows=1, cols=1)
    status = "Success" if success else "Failed"
    color = "green" if success else "red"
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=1 if success else 0,
        title={"text": f"Registration {status}"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": color}}
    ))
    st.plotly_chart(fig)

def main():
    st.title("Face Registration System")
    st.write("Please capture your face image below.")
    
    image = st.camera_input("Take a Picture")
    
    if image:
        face_encoding = capture_face_encodings(image)
        if face_encoding is not None:
            st.success("Face captured successfully! Please enter your details.")
            
            name = st.text_input("Name")
            reg_no = st.text_input("Registration Number")
            mobile_no = st.text_input("Mobile Number")
            year = st.selectbox("Select Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
            branch = st.selectbox("Select Branch", ["ECE", "CSE", "CSE DS", "CSE AIML"])
            section = st.selectbox("Select Section", ["A", "B", "C", "D", "E"])
            
            if st.button("Submit"):
                new_entry = pd.DataFrame({
                    "Name": [name],
                    "Registration Number": [reg_no],
                    "Mobile Number": [mobile_no],
                    "Year": [year],
                    "Branch": [branch],
                    "Section": [section],
                    "Encoding": [np.array2string(face_encoding, separator=',')]
                })
                
                global df
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_excel(data_file, index=False)
                st.success("Details recorded successfully!")
                plot_registration_status(True)
        else:
            st.error("Failed to capture face. Please try again.")
            plot_registration_status(False)

if __name__ == "__main__":
    main()
