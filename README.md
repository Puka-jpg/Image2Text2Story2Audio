# Image2Text2Story2Audio

# Image to Audio Story Generator

A Python application that transforms images into creative audio stories using AI technology. This project combines image captioning, story generation, and text-to-speech conversion to create engaging audio narratives from uploaded images.

## 🚀 Features

- Image to text conversion using Salesforce BLIP model
- Story generation using Google's Gemini Pro AI
- Text-to-speech conversion with gTTS
- User-friendly web interface built with Streamlit

## 🛠️ Technologies Used

- Python 3.8+
- Streamlit for web interface
- Hugging Face Transformers
- Google Gemini Pro API
- gTTS (Google Text-to-Speech)

## 🎯 How to Use

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your environment variables in `.env`
4. Run: `streamlit run main.py`
5. Upload an image and watch the magic happen!

## Usage Flow

1. User uploads an image through Streamlit interface
2. Image is processed to generate a descriptive caption
3. Caption is used to generate a creative story
4. Story is converted to speech
5. Results are displayed in expandable sections
6. Audio can be played directly in the browser

## Error Handling

- Each major function includes try-except blocks
- Errors are logged with descriptive messages
- Graceful fallbacks when processing fails

## File Management

- Handles temporary file storage for uploads
- Manages audio file creation and storage
- Cleans up temporary files

## 🤝 Contributing

Feel free to fork, create a pull request, or open issues.

## 👨‍💻 Author

Pukar Kafle

## 🙏 Acknowledgments

- Salesforce BLIP model for image captioning
- Google Gemini Pro for story generation
- Streamlit for the awesome web framework
