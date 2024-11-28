from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import os
import google.generativeai as genai
import requests
from gtts import gTTS
import warnings
import logging
from absl import logging as absl_logging
import streamlit as st

warnings.filterwarnings('ignore')
absl_logging.set_verbosity(absl_logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('transformers').setLevel(logging.ERROR)

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def img2text(url):
    try:
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        text = image_to_text(url)[0]["generated_text"]
        print("Image caption:", text)
        return text
    except Exception as e:
        print(f"Error in image to text conversion: {str(e)}")
        return None

def generate_story(scenario):
    if not scenario:
        return "Could not generate story due to missing scenario."
    
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""You are a creative storyteller. 
        Generate a short, engaging story based on this scenario in exactly 20 words: {scenario}
        The story should be imaginative and fun."""
        
        response = model.generate_content(prompt)
        story = response.text
        print("Generated story:", story)
        return story
        
    except Exception as e:
        print(f"Error in story generation: {str(e)}")
        return None

print("Current working directory:", os.getcwd())

image_path = "family.webp"
scenario = img2text(image_path)
if scenario:
    story = generate_story(scenario)

def text2speech(message):
    try:
        print("Converting story to speech...")
        
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, "story_audio.mp3")
        
        tts = gTTS(text=message, lang='en')
        tts.save(file_path)
        
        if os.path.exists(file_path):
            print(f"Audio file created successfully at: {file_path}")
        else:
            print("File was not created!")
        return True
    except Exception as e:
        print(f"Error in text to speech conversion: {str(e)}")
        return False

if scenario and story:
    text2speech(story)

def main():
    st.set_page_config(page_title="Image to audio story", page_icon="", layout="wide")
    
    st.header("Turn img into audio story")
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        print(uploaded_file)
        
        bytes_data = uploaded_file.getvalue()
        
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        
        with st.expander("story"):
            st.write(story)
        
        st.audio("story_audio.mp3")

if __name__ == "__main__":
    main()
