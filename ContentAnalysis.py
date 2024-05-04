'''
!pip install git+https://github.com/openai/whisper.git

!sudo apt update && !sudo apt install ffmpeg -y

!ffmpeg -version

!pip install openai

'''
import whisperx
import os
import csv
import openai
from openai import OpenAI
import json
import time

def extract_and_convert_json(s):
    # Find the leftmost opening curly brace
    start_index = s.find('{')
    # Find the rightmost closing curly brace
    end_index = s.rfind('}')

    # Check if both curly braces are found
    if start_index == -1 or end_index == -1:
        print("No valid JSON format found.")
        return None

    # Extract the substring that is presumed to be JSON
    json_string = s[start_index:end_index + 1]

    try:
        # Convert the JSON string to a Python dictionary
        json_data = json.loads(json_string)
        return json_data
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None
class OpenAIIntegration:
    def __init__(self, key):
        self.key= key
        self.client = OpenAI(api_key=key)

        self.device = "cuda"
        self.batch_size = 16  # Adjust based on your GPU memory
        self.compute_type = "float16"  # Use "int8" for lower memory usage, with potential accuracy trade-off
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)

    def transcribe_audio(self, audio_path):
        audio = whisperx.load_audio(self.audio_path)
        transcription_result = self.model.transcribe(audio, batch_size=self.batch_size)
        return transcription_result

    def get_chat_response(self, transcribed_text, topics, question, role="user"):
        system_message = "You are a job interview anaysis bot."

        prompt = f"""
        System Instructions:
        - Analyze the transcribed interview response below.
        - Evaluate the response against predefined topics.
        - For each topic, assign a score based on the completeness of the answer:
        - 0: Not answered
        - 1: Partially answered
        - 2: Fully answered

        Output the results in JSON format with two key components:
        1. "topic_points": a dictionary with each topic and its corresponding score
        2. "summary": a brief summary of the overall response

        1. Here is the interview question posed to the candidate: {question}
        2. Below is the interviewee's response:

        Transcribed Interview Response:
        {transcribed_text}

        Assessment Criteria:
        Here are the topics that should be answered by the interviewee:
        {topics}
        """

        # specified model and messages
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": role, "content": prompt}
            ],
            temperature = 0.2, #deterministic
        )

        # Return the response content
        return extract_and_convert_json(completion.choices[0].message.content)



if __name__ == "main":
    # Configuration and file paths
    directory_path = "/content/transcription.txt"
    audio_file = "/content/P1.wav"

    openai_integration = OpenAIIntegration()

    # Transcribe the audio file
    transcription = openai_integration.transcribe_audio(audio_file)

    # Example JSON for topics (replace with actual JSON input as required)
    question = "Tell me about yourself"

    topics = """
    1- Personal background
    2- Educational background
    3- Why they are interested in the job/industry
    4- Curious
    5- Clothing
    6- Interested in sports
    7- Enjoys Problem Solving
    """

    # Get response from GPT-4 based on the transcription and topics
    response = openai_integration.get_chat_response(transcription, topics, question)
    print(response)
