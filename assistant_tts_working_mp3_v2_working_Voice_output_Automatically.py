import wave
from groq import Groq
from PIL import ImageGrab, Image
import google.generativeai as genai
import pyperclip
import cv2
import edge_tts  # Import the Edge TTS library
from pydub import AudioSegment
import pyaudio
import asyncio
import threading
import os

groq_client = Groq(api_key="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")
genai.configure(api_key='AIzaSyAJyVbIvw9xDpCM3pCbE7WfdIjNk9KQzWo')
web_cam = cv2.VideoCapture(0)

sys_msg = (
    'You are a multi-modal AI voice assistant. Your name is  "Isha," Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)


# sys_msg = (
#     'You are a multi-modal AI voice assistant. Your name is  "Vision-Chat"' 
#     'Your user may or may not have attached a photo for context '
#     '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
#     'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
#     'factual response possible, carefully considering all previous generated text in your response before '
#     'adding new tokens to the response. Do not expect or request images, just use the context if added. '
#     'Use all of the context of this conversation so your response is relevant to the conversation. Make '
#     'your responses clear and concise, avoiding any verbosity.' 
# )



convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-2.0-flash-latest', 
                              generation_config=generation_config)
                            #   safety_settings=safety_settings)

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]

    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

def web_cam_capture():
    if not web_cam.isOpened():
        print("Error: Camera did not open successfully")
        exit()
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text




def detect_image(image_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)
    if os.path.exists(image_path):
        return image_path
    else:
        print(f"Error: Image '{image_name}' not found in the current directory.")
        return None
# "detect the image-- myphoto.jpg - What objects are in this image?"
# "detect the image-- myphoto.jpg - What objects are in this image?"
# "detect the image-- myphoto.jpg - What objects are in this image?"



async def read_response_aloud(text):
    communicate = edge_tts.Communicate(text, "en-US-AvaMultilingualNeural")
    await communicate.save("response.mp3")

    # Convert MP3 to WAV
    mp3_audio = AudioSegment.from_mp3("response.mp3")
    wav_path = "response.wav"
    mp3_audio.export(wav_path, format="wav")

    # Play the WAV file using wave module
    CHUNK = 1024
    wf = wave.open(wav_path, 'rb')
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read data
    data = wf.readframes(CHUNK)

    # Play stream
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    # Stop stream
    stream.stop_stream()
    stream.close()

    # Close PyAudio
    p.terminate()

def read_response_and_play(text):
    asyncio.run(read_response_aloud(text))

while True:
    prompt = input('USER: ')


    if prompt.startswith('detect the image--'):
        # Extract image name and user question
        parts = prompt.split('--', 1)
        if len(parts) == 2:
            image_name = parts[1].split('-')[0].strip()
            user_question = parts[1].split('-', 1)[1].strip() if '-' in parts[1] else ""
            
            image_path = detect_image(image_name)
            if image_path:
                visual_context = vision_prompt(prompt=user_question, photo_path=image_path)
                response = groq_prompt(prompt=user_question, img_context=visual_context)
                print(response)
                
                # Start the text-to-speech in a separate thread
                tts_thread = threading.Thread(target=read_response_and_play, args=(response,))
                tts_thread.start()
            continue    


# "detect the image-- myphoto.jpg - What objects are in this image?"
# "detect the image-- myphoto.jpg - What objects are in this image?"
# "detect the image-- myphoto.jpg - What objects are in this image?"




    call = function_call(prompt)

    if 'take screenshot' in call:
        print('Take Screenshot')
        take_screenshot()
        visual_context = vision_prompt(prompt=prompt, photo_path='screenshot.jpg')
    elif 'capture webcam' in call:
        print('Capturing webcam')
        web_cam_capture()
        visual_context = vision_prompt(prompt=prompt, photo_path='webcam.jpg')
    elif 'extract_clipboard' in call:
        print('Copying clipboard text')
        paste = get_clipboard_text()
        prompt = f'{prompt}\n\n CLIPBOARD CONTENT: {paste}'
        visual_context = None
    else:
        visual_context = None

    response = groq_prompt(prompt=prompt, img_context=visual_context)
    print(response)

    # Start the text-to-speech in a separate thread
    tts_thread = threading.Thread(target=read_response_and_play, args=(response,))
    tts_thread.start()


