import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from skimage.transform import resize
import plotly.graph_objects as go
import plotly.express as px
import base64
import os

# Define a function to load the model
#@st.cache_resource(show_spinner=False)
#def load_model_from_path(model_path):
#    try:
#        model = load_model(model_path)
#        return model
#    except Exception as e:
#        st.error(f"Error loading model: {e}")
#        return None
#
## Path to the model file
#model_path = './models/re-trained_model.h5'
#
## Check if the model file exists
#if os.path.exists(model_path):
#    model = load_model_from_path(model_path)
#    if model:
#        st.success("Model loaded successfully.")
#else:
#    st.error("Model file does not exist.")

model = load_model('./models/re-trained_model.h5')

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to load and preprocess the audio file
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# Function to predict and visualize the results
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    mean_probabilities = np.mean(y_pred, axis=0)
    sorted_indices = np.argsort(mean_probabilities)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_probabilities = mean_probabilities[sorted_indices]

    # Assign a color scheme that works well with white text
    colors = px.colors.qualitative.Bold

    sorted_colors = [colors[i % len(colors)] for i in sorted_indices]

    # Display the prediction results as a horizontal bar chart using Plotly
    fig = go.Figure(go.Bar(
        x=sorted_probabilities,
        y=sorted_classes,
        orientation='h',
        marker=dict(color=sorted_colors),
        text=[f'{prob:.2%}' for prob in sorted_probabilities],
        textposition='auto',
        textfont=dict(color='white')  # Set text color to white
    ))

    fig.update_layout(
        title='Category Probability Distribution',
        xaxis_title='Probability Percentage',
        yaxis_title='Category',
        yaxis=dict(autorange='reversed')
    )

    st.plotly_chart(fig)

    return sorted_classes[0]

# Streamlit App
st.title("🎶 Music Genre Classifier")
st.markdown('To develop an application that classifies music into different genres using deep learning algorithms, providing users with genre predictions based on audio features.')

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload an audio file and the model will predict the most likely genre", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    # Preprocess the uploaded audio file
    X_test = load_and_preprocess_data(uploaded_file)

    # Perform prediction and display the result
    predicted_category = model_prediction(X_test)
    st.success(f"The most likely category: **{predicted_category}**")

else:
    sample_data = st.checkbox('Use a sample data')
    if sample_data:
        # Load the sample data
        sample_path = '.\data\The Neighbourhood - Sweater Weather.mp3'
        st.audio(sample_path, format='audio/mp3')
        st.markdown('''**Credit:** The sample audio is taken from [The Neighbourhood - Sweater Weather](https://www.youtube.com/watch?v=GCdwKhTtNNw&list=RDMM&start_radio=1)''')
        X_sample= load_and_preprocess_data(sample_path)
        predicted_category_sample = model_prediction(X_sample)
        st.success(f"The most likely category: **{predicted_category_sample}**")

# Base directory for assets
assets_folder = "assets"

def image_to_base64(image_path):
    """Convert an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Load and convert images to base64
linkedin_logo_base64 = image_to_base64(os.path.join(assets_folder, 'linkedin-logo.png'))
medium_logo_base64 = image_to_base64(os.path.join(assets_folder, 'medium-logo.png'))
gmail_logo_base64 = image_to_base64(os.path.join(assets_folder, 'gmail-logo.png'))

# Footer HTML
footer = f"""
<div style="display: flex; justify-content: center; align-items: center; padding: 10px 0;">
    <a href="https://www.linkedin.com/in/muhammadiqbalanwar/" target="_blank">
        <img src="data:image/png;base64,{linkedin_logo_base64}" alt="LinkedIn" style="height: 30px; width: 30px; margin-right: 10px;"/>
    </a>
    <a href="https://medium.com/@helloitsiqbal" target="_blank">
        <img src="data:image/png;base64,{medium_logo_base64}" alt="Medium" style="height: 30px; width: 30px; margin-right: 10px;"/>
    </a>
    <a href="mailto:iqbala.work@gmail.com">
        <img src="data:image/png;base64,{gmail_logo_base64}" alt="Gmail" style="height: 30px; width: 30px;"/>
    </a>
    <div style="margin-left: 20px;">&copy; Muhammad <strong>Iqbal</strong> A. | 2024</div>
</div>
"""

# Display footer
st.divider()
st.markdown('''
I am **open to side hustle opportunities** related to machine learning, forecasting, optimization, & deep learning. 
If you have a project or need assistance in these areas, feel free to reach out to me via email or LinkedIn 😉

Talk to you soon! 👋
'''
)
st.markdown(footer, unsafe_allow_html=True)
