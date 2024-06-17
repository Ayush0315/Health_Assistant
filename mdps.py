import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from pdf2image import convert_from_path
import pytesseract
import cv2
from PIL import Image
from geopy.geocoders import Nominatim
import requests
from twilio.rest import Client 
import numpy as np

# Set Tesseract OCR path (update this to your installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Load models
models_path = 'C:/Users/Patil/OneDrive/Desktop/Ayush/Project/Saved Models/'
diabetes_model = pickle.load(open(os.path.join(models_path, 'diabetes_model.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(models_path, 'heart_disease_model.sav'), 'rb'))

st.set_page_config(page_title="Health Assistant", page_icon="⚕️")
# Sidebar menu
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Health Assistant</h2>", unsafe_allow_html=True)
    selected = option_menu('', 
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Graphs', 'Find Nearest Hospital'], 
                           menu_icon='hospital-fill', 
                           icons=['activity', 'heart', 'file-bar-graph', 'map'], 
                           default_index=0, 
                           styles={
                               "container": {"padding": "0!important", "background-color": "#2e7bcf"},
                               "icon": {"color": "white", "font-size": "25px"}, 
                               "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#FF4B4B"},
                               "nav-link-selected": {"background-color": "#FF4B4B"}
                           })

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf(pdf_file):
    """Convert PDF to images and extract text from each page."""
    images = convert_from_path(pdf_file)
    text = ''
    for image in images:
        text += extract_text_from_image(image)
    return text

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and extract text."""
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.png', '.jpg', '.jpeg']:
            image = Image.open(uploaded_file)
            return extract_text_from_image(image)
        elif file_extension == '.pdf':
            with open("temp_file.pdf", "wb") as f:
                f.write(uploaded_file.read())
            return extract_text_from_pdf("temp_file.pdf")
    return None

def parse_extracted_text(text):
    """Parse the extracted text to get input values."""
    input_values = {}
    lines = text.split('\n')
    for line in lines:
        key_value = line.split(':')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            try:
                if key in ['Serum Creatinine (SC)', 'LDL', 'Fasting Blood Sugar (FBS)', 
                           'Post Prandial Blood Sugar (PPBS)', 'Cholesterol', 'VLDL', 
                           'RBC Count', 'Triglycerides', 'Haemoglobin', 'Platelet Count (in lakhs)', 'ESR']:
                    input_values[key] = float(value)
                else:
                    input_values[key] = value
            except ValueError:
                input_values[key] = value
    return input_values

def get_nearby_hospitals(location, radius=2500):
    """Fetch nearby hospitals using Nominatim and Overpass API."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{location[0]},{location[1]});
      way["amenity"="hospital"](around:{radius},{location[0]},{location[1]});
      relation["amenity"="hospital"](around:{radius},{location[0]},{location[1]});
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    hospitals = []
    for element in data['elements']:
        lat = element.get('lat') or element['center']['lat']
        lon = element.get('lon') or element['center']['lon']
        hospital = {
            'name': element.get('tags', {}).get('name', 'Unknown'),
            'lat': lat,
            'lon': lon,
            'phone': element.get('tags', {}).get('phone', 'N/A')  # Get phone number if available
        }
        hospitals.append(hospital)
    return hospitals

def get_current_location(address):
    geolocator = Nominatim(user_agent="health_app")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

def find_nearest_hospital():
    st.title('Find Nearest Hospital and Book Appointment')

    address = st.text_input("Enter your location to find nearby hospitals (e.g., '1600 Amphitheatre Parkway, Mountain View, CA' or '37.422, -122.084')")

    if st.button('Find Nearest Hospital'):
        if address:
            location = get_current_location(address)
            if location:
                hospitals = get_nearby_hospitals(location)
                st.write(f"Debug: Number of hospitals found: {len(hospitals)}")
                
                if not hospitals:
                    st.warning("No hospitals found within the specified radius.")
                    return
                
                st.map(pd.DataFrame(hospitals, columns=['lat', 'lon']))

                hospital_names = [hospital['name'] for hospital in hospitals]
                selected_hospital = st.selectbox('Select Hospital', hospital_names)

                hospital_details = next(hospital for hospital in hospitals if hospital['name'] == selected_hospital)
                st.write(f"**Name**: {hospital_details['name']}")
                st.write(f"**Phone**: {hospital_details['phone']}")

                if st.button('Book Appointment'):
                    call_sid = book_appointment(hospital_details['phone'])
                    st.success(f"Appointment booked! Call SID: {call_sid}")
            else:
                st.error("Could not find the location. Please enter a valid address.")
        else:
            st.warning("Please enter a location.")

def book_appointment(hospital_phone_number):
    twilio_account_sid = "AC03756e9e6a51d3ce3723f207707a02ad"
    twilio_auth_token = "ebc11770459aa33f223358af7aa143b4"
    twilio_phone_number = "+917795768427"
    
    client = Client(twilio_account_sid, twilio_auth_token)
    
    call = client.calls.create(
        to=hospital_phone_number,
        from_=twilio_phone_number,
        url="http://demo.twilio.com/docs/voice.xml"  # Replace with your own URL for booking message
    )
    
    return call.sid
def find_nearest_hospital():
    st.title('Find Nearest Hospital and Book Appointment')
    
    address = st.text_input("Enter your location to find nearby hospitals")

    if st.button('Find Nearest Hospital'):
        if address:
            location = get_current_location(address)
            if location:
                hospitals = get_nearby_hospitals(location)
                
                if not hospitals:
                    st.warning("No hospitals found within the specified radius.")
                    return
                
                st.map(pd.DataFrame(hospitals, columns=['lat', 'lon']))
        
                hospital_names = [hospital['name'] for hospital in hospitals]
                selected_hospital = st.selectbox('Select Hospital', hospital_names)
        
                hospital_details = next(hospital for hospital in hospitals if hospital['name'] == selected_hospital)
                st.write(f"**Name**: {hospital_details['name']}")
                st.write(f"**Phone**: {hospital_details['phone']}")
        
                if st.button('Book Appointment'):
                    call_sid = book_appointment(hospital_details['phone'])
                    st.success(f"Appointment booked! Call SID: {call_sid}")
            else:
                st.error("Could not find the location. Please enter a valid address.")
        else:
            st.warning("Please enter a location.")

def diabetes_prediction(uploaded_file=None):
    st.title('Diabetes Prediction using ML')

    # Define the ranges for diabetes determination
    fbs_normal_range = (70, 110)
    ppbs_normal_range = (70, 150)

    if uploaded_file:
        extracted_text = process_uploaded_file(uploaded_file)
        st.text_area("Extracted Text", extracted_text, height=200)
        user_input = parse_extracted_text(extracted_text)
    else:
        user_input = {
            'SC': st.slider('Serum Creatinine (SC)', min_value=0.6, max_value=1.4, step=0.1),
            'LDL': st.slider('LDL', min_value=90, max_value=159, step=10),
            'Fasting Blood Sugar (FBS)': st.slider('Fasting Blood Sugar (FBS)', min_value=70, max_value=200, step=10),
            'Post Prandial Blood Sugar (PPBS)': st.slider('Post Prandial Blood Sugar (PPBS)', min_value=70, max_value=310, step=10),
            'Cholesterol': st.slider('Cholesterol', min_value=180, max_value=240, step=5),
            'VLDL': st.slider('VLDL', min_value=10, max_value=40, step=10),
            'RBC Count': st.slider('RBC Count', min_value=3, max_value=6, step=1),
            'Triglycerides': st.slider('Triglycerides', min_value=120, max_value=240, step=20),
            'Haemoglobin': st.slider('Haemoglobin', min_value=11.5, max_value=16.0, step=0.5),
            'Platelet Count (in lakhs)': st.slider('Platelet Count (in lakhs)', min_value=1.5, max_value=4.0, step=0.1),
            'ESR': st.slider('ESR', min_value=0, max_value=100, step=1)
        }

    if st.button('Diabetes Test Result'):
        fbs_value = user_input.get('Fasting Blood Sugar (FBS)')
        ppbs_value = user_input.get('Post Prandial Blood Sugar (PPBS)')

        if (fbs_value is not None and ppbs_value is not None):
            if fbs_value > fbs_normal_range[1] or ppbs_value > ppbs_normal_range[1]:
                prediction = 1  # Person is diabetic
            else:
                prediction = 0  # Person is not diabetic
        else:
            user_input_values = np.array(list(user_input.values())).reshape(1, -1)
            prediction = diabetes_model.predict(user_input_values)[0]

        if prediction == 1:
            st.markdown(
                """
                <div style="padding: 10px; border-radius: 5px; background-color: red; color: white; font-size: large;">
                    The person is diabetic
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("<h2>Prescription:</h2>", unsafe_allow_html=True)
            st.write("<ul>", unsafe_allow_html=True)
            st.write("<li><strong>Medication 1:</strong> Metformin - Take 1 tablet (500 mg) twice daily, with breakfast and dinner.</li>", unsafe_allow_html=True)
            st.write("<li><strong>Medication 2:</strong> Glipizide - Take 1 tablet (5 mg) once daily, in the morning.</li>", unsafe_allow_html=True)
            st.write("<li><strong>Lifestyle changes:</strong> Incorporate diet and exercise changes to manage blood sugar levels effectively.</li>", unsafe_allow_html=True)
            st.write("</ul>", unsafe_allow_html=True)

            st.write('<h4> Please visit the Find Nearest Hospital page </h4>', unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div style="padding: 10px; border-radius: 5px; background-color: green; color: white; font-size: large;">
                    The person is not diabetic
                </div>
                """,
                unsafe_allow_html=True
            )

def heart_disease_prediction(uploaded_file=None):
    st.title('Heart Disease Prediction using ML')

    user_input = {}
    direct_prediction = False

    if uploaded_file is not None:
        extracted_text = process_uploaded_file(uploaded_file)
        st.text_area("Extracted Text", extracted_text, height=200)
        
        if "Heart Disease: 1" in extracted_text:
            direct_prediction = True
        else:
            user_input = parse_extracted_text(extracted_text)

    if not user_input:
        user_input = {
            'Age': st.slider('Age', min_value=0, max_value=100, step=1),
            'Sex': st.selectbox('Sex', ['Male', 'Female']),
            'Chest Pain types': st.slider('Chest Pain types', min_value=0, max_value=3, step=1),
            'Resting Blood Pressure': st.slider('Resting Blood Pressure', min_value=90, max_value=200, step=1),
            'Serum Cholesterol (mg/dl)': st.slider('Serum Cholesterol (mg/dl)', min_value=100, max_value=400, step=1),
            'Fasting Blood Sugar > 120 mg/dl': st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes']),
            'Resting Electrocardiographic results': st.slider('Resting Electrocardiographic results', min_value=0, max_value=2, step=1),
            'Maximum Heart Rate achieved': st.slider('Maximum Heart Rate achieved', min_value=60, max_value=200, step=1),
            'Exercise Induced Angina': st.selectbox('Exercise Induced Angina', ['No', 'Yes']),
            'ST depression induced by exercise': st.slider('ST depression induced by exercise', min_value=0.0, max_value=6.2, step=0.1),
            'Slope of the peak exercise ST segment': st.slider('Slope of the peak exercise ST segment', min_value=0, max_value=2, step=1),
            'Major vessels colored by fluoroscopy': st.slider('Major vessels colored by fluoroscopy', min_value=0, max_value=4, step=1),
            'Thalassemia': st.slider('Thalassemia', min_value=0, max_value=2, step=1)
        }

    if st.button('Heart Disease Test Result'):
        if direct_prediction:
            prediction = 1
        else:
            # Convert categorical variables to numerical values
            user_input['Sex'] = 1 if user_input['Sex'] == 'Male' else 0
            user_input['Fasting Blood Sugar > 120 mg/dl'] = 1 if user_input['Fasting Blood Sugar > 120 mg/dl'] == 'Yes' else 0
            user_input['Exercise Induced Angina'] = 1 if user_input['Exercise Induced Angina'] == 'Yes' else 0

            # Ensure the input order matches the model's expected feature order
            user_input_values = [
                user_input['Age'],
                user_input['Sex'],
                user_input['Chest Pain types'],
                user_input['Resting Blood Pressure'],
                user_input['Serum Cholesterol (mg/dl)'],
                user_input['Fasting Blood Sugar > 120 mg/dl'],
                user_input['Resting Electrocardiographic results'],
                user_input['Maximum Heart Rate achieved'],
                user_input['Exercise Induced Angina'],
                user_input['ST depression induced by exercise'],
                user_input['Slope of the peak exercise ST segment'],
                user_input['Major vessels colored by fluoroscopy'],
                user_input['Thalassemia']
            ]

            # Convert all input values to numeric types (floats)
            user_input_values = [float(value) for value in user_input_values]
            user_input_values = np.array(user_input_values).reshape(1, -1)

            prediction = heart_disease_model.predict(user_input_values)[0]

        if prediction == 1:
            st.markdown(
                """
                <div style="padding: 10px; border-radius: 5px; background-color: red; color: white; font-size: large;">
                    The person has heart disease
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("<h2>Prescription:</h2>", unsafe_allow_html=True)
            st.write("<ul>", unsafe_allow_html=True)
            st.write("<li><strong>Medication 1:</strong> Aspirin - Take 1 tablet (81 mg) daily with food.</li>", unsafe_allow_html=True)
            st.write("<li><strong>Medication 2:</strong> Beta-blockers - Take 1 tablet (25 mg) twice daily, in the morning and evening.</li>", unsafe_allow_html=True)
            st.write("<li><strong>Medication 3:</strong> ACE inhibitors - Take 1 tablet (10 mg) once daily in the morning.</li>", unsafe_allow_html=True)
            st.write("</ul>", unsafe_allow_html=True)

            st.write('<h4> Please visit the Find Nearest Hospital page </h4>', unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div style="padding: 10px; border-radius: 5px; background-color: green; color: white; font-size: large;">
                    The person does not have heart disease
                </div>
                """,
                unsafe_allow_html=True
            )

def show_graphs():
    st.title('Graphs and Visualizations')
    
    # Sample data
    df = pd.DataFrame({
        'Disease': ['Diabetes', 'Heart Disease'],
        'Cases': [200, 150],
        'Deaths': [50, 80],
        'Recovered': [120, 50]
    })
    
    # Bar chart for Disease Cases
    fig = px.bar(df, x='Disease', y='Cases', title='Disease Cases')
    st.plotly_chart(fig)
    
    # Pie chart for Deaths by Disease
    fig = px.pie(df, values='Deaths', names='Disease', title='Deaths by Disease')
    st.plotly_chart(fig)
    
    # Line chart for Cases, Deaths, and Recovered
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Disease'], y=df['Cases'], mode='lines+markers', name='Cases'))
    fig.add_trace(go.Scatter(x=df['Disease'], y=df['Deaths'], mode='lines+markers', name='Deaths'))
    fig.add_trace(go.Scatter(x=df['Disease'], y=df['Recovered'], mode='lines+markers', name='Recovered'))
    fig.update_layout(title='Disease Cases, Deaths, and Recovered', xaxis_title='Disease', yaxis_title='Count')
    st.plotly_chart(fig)
    
    # Scatter plot for Cases vs Deaths
    fig = px.scatter(df, x='Cases', y='Deaths', color='Disease', size='Cases', hover_name='Disease', title='Cases vs Deaths')
    st.plotly_chart(fig)

    # Box plot for Cases, Deaths, and Recovered
    df_melted = df.melt(id_vars=['Disease'], value_vars=['Cases', 'Deaths', 'Recovered'], var_name='Metric', value_name='Value')
    fig = px.box(df_melted, x='Metric', y='Value', color='Disease', title='Box Plot of Cases, Deaths, and Recovered')
    st.plotly_chart(fig)
    
    # Heatmap for Cases, Deaths, and Recovered
    df_heatmap = df.set_index('Disease').transpose()
    fig = go.Figure(data=go.Heatmap(
        z=df_heatmap.values,
        x=df_heatmap.columns,
        y=df_heatmap.index,
        colorscale='Viridis'))
    fig.update_layout(title='Heatmap of Cases, Deaths, and Recovered')
    st.plotly_chart(fig)

if selected == 'Diabetes Prediction':
    diabetes_prediction(st.file_uploader("Upload PDF or Image file", type=['pdf', 'png', 'jpg', 'jpeg']))
elif selected == 'Heart Disease Prediction':
    heart_disease_prediction(st.file_uploader("Upload PDF file", type=['pdf']))
elif selected == 'Find Nearest Hospital':
    find_nearest_hospital()
elif selected == 'Graphs':
    show_graphs()
    
