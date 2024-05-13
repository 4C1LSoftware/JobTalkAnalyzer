import os
from pydub import AudioSegment
from moviepy.editor import concatenate_videoclips, VideoFileClip
from app.analyzer.JobTalkAnalyzer import JobTalkAnalyzer
from app.analyzer.ContentAnalysis import OpenAIIntegration
from app.analyzer.config import OPENAI_API_KEY
import json
from time import sleep
from dotenv import load_dotenv
from app.codec_converter import convert_webm_to_wav, convert_webm_to_mp4

load_dotenv()

def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000  # Convert from milliseconds to seconds

class Analyzer:
    def __init__(self, path: str, pid: int, candidate_id:int, interview_id:int, interview_guide: {}):
        self.path = path
        self.wav_files = []
        self.mp4_files = []
        self._validate_files()
        self.pid= pid
        self.openapi_key= os.getenv("OPENAI_API_KEY")
        self.candidate_id = candidate_id
        self.interview_id = interview_id
        self.interview_guide= interview_guide
        self.base_path= f"analysis/{self.candidate_id}/{self.interview_id}/"
        print(self.base_path)
        self.dump_path = os.path.join(self.base_path, self.pid)
        print(self.dump_path)

    def _validate_files(self):
        # List all files in the directory
        files = os.listdir(self.path)
        webm_files = [f for f in files if f.endswith('.webm')]
        for webm in webm_files:
            filename_wo_extension, _ = os.path.splitext(webm)
            filepath = os.path.join(self.path, filename_wo_extension)
            convert_webm_to_wav(filepath)
            convert_webm_to_mp4(filepath)
        # Filter out .wav and .mp4 files
        files = os.listdir(self.path)
        self.wav_files = [f for f in files if f.endswith('.wav')]
        self.mp4_files = [f for f in files if f.endswith('.mp4')]
        print(self.wav_files)
        print(self.mp4_files)

        # Check for exactly 5 .wav and 5 .mp4 files
        assert len(self.wav_files) == len(self.mp4_files), "There number of .wav and .mp4 should be equal"

    def combine_wav_files(self):
        # Combine all .wav files into one
        combined = AudioSegment.from_file(os.path.join(self.path, self.wav_files[0]))
        for wav_file in self.wav_files[1:]:
            audio = AudioSegment.from_file(os.path.join(self.path, wav_file))
            combined += audio
        combined.export(os.path.join(self.path, "combined_audio.wav"), format='wav')

    def combine_mp4_files(self):
        # Combine all .mp4 files into one
        clips = [VideoFileClip(os.path.join(self.path, mp4_file)) for mp4_file in self.mp4_files]
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(os.path.join(self.path, "combined_video.mp4"), codec="libx264")

    def performance_analysis(self):
        audio_path= os.path.join(self.path, "combined_audio.wav")
        video_path= os.path.join(self.path, "combined_video.mp4")
        analyzer= JobTalkAnalyzer(self.pid, audio_path, video_path, self.candidate_id, self.interview_id)
        analyzer.analyze()

        analyzer.predict_scores()
        
    def content_analysis(self):

        content_analysis_list = []
        combined_timestamped_transcription = []
        last_end_time = 0  # Initialize the last timestamp

        for filename in self.wav_files:
            filename_wo_extension, _ = os.path.splitext(filename)
            interview_id, q_id = filename_wo_extension.split("_")
            if q_id not in self.interview_guide:
                print(f"ERROR: Interview {q_id} not in {self.interview_guide.keys()}")
                return False
            interview_guide_item = self.interview_guide[q_id]
            interview_question = interview_guide_item["question"]
            keyword_list = interview_guide_item["keyword_list"]

            keywords_str = ""
            for i, keyword in enumerate(keyword_list):
                keywords_str += f"{i+1}- {keyword} "

            openai_integration = OpenAIIntegration(self.openapi_key)
            file_path = os.path.join(self.path, filename)
            # Transcribe the audio file
            transcription, timestamped_transcription = openai_integration.transcribe_audio(file_path)

            # Adjust timestamps and append to combined transcription
            adjusted_transcription = []
            for word_info in timestamped_transcription:
                if 'start' in word_info and 'end' in word_info:
                    word_info['start'] += last_end_time
                    word_info['end'] += last_end_time
                    adjusted_transcription.append(word_info)
            
            combined_timestamped_transcription.extend(adjusted_transcription)

            audio_duration = get_audio_duration(file_path)
            last_end_time += audio_duration  # Update last_end_time to include potential silence
            
            response = openai_integration.get_chat_response(transcription, keywords_str, interview_question, question_id= q_id)
            content_analysis_list.append(response)

        # After processing all files, dump the combined transcription to a JSON file
        os.makedirs(self.dump_path, exist_ok=True)
        output_file_path = os.path.join(self.dump_path, "transcription.json")
        with open(output_file_path, 'w') as json_file:
            json.dump(combined_timestamped_transcription, json_file, indent=4)

        file_path= os.path.join(self.dump_path, "content_analysis.json")

        # Write the dictionary to a file in JSON format
        with open(file_path, 'w') as file:
            json.dump(content_analysis_list, file, indent=4)

        return True
    
    def finish_analysis(self):
        # Define the paths to the two JSON files
        content_analysis_path = os.path.join(self.dump_path, 'content_analysis.json')
        performance_analysis_path = os.path.join(self.dump_path, 'performance_analysis.json')
        
        # Read the content from content_analysis.json
        with open(content_analysis_path, 'r') as file:
            content_analysis = json.load(file)
        
        # Read the content from performance_analysis.json
        with open(performance_analysis_path, 'r') as file:
            performance_analysis = json.load(file)
        
        # Combine the read content into a new dictionary
        combined_analysis = {
            'content_analysis': content_analysis,
            'performance_analysis': performance_analysis
        }
        
        # Define the path for the new combined JSON file
        combined_analysis_path = os.path.join(self.dump_path, 'analysis.json')
        
        # Write the combined content to analysis.json
        with open(combined_analysis_path, 'w') as file:
            json.dump(combined_analysis, file, indent=4)  # Use indent for pretty-printing

        print(f"Analysis finished. Combined file saved to {combined_analysis_path}")

    def analyze(self):
        content_success=self.content_analysis()
        if not content_success:
            print("Content analysis failed exitting...")
            return False

        self.combine_wav_files()
        self.combine_mp4_files()
        self.performance_analysis()
        self.finish_analysis()
        return True

interview_guide= {
    "3": {
        "question" : "Tell me about yourself",
        "keyword_list": [
            "Personal background",
            "Educational background",
            "Why they are interested in the job/industry",
            "Curious",
            "Clothing",
            "Interested in sports",
            "Dislikes team work"
        ]
    },
    "43": {
        "question" : "Tell me about your experience working in a team",
        "keyword_list": [
            "Personal background",
            "Educational background",
            "Why they are interested in the job/industry",
            "Is a selfish person",
            "Is an arrogant person",
            "Interested in sports",
            "Likes team work"
        ]
    },
    "5": {
        "question" : "Tell me about your experience working in a team",
        "keyword_list": [
            "Personal background",
            "Educational background",
            "Why they are interested in the job/industry",
            "Is a selfish person",
            "Is an arrogant person",
            "Interested in sports",
            "Likes team work"
        ]
    }
}
# Example usage
'''print(os.getcwd())
analyzer = Analyzer('../../recordings/88/', "22", "2" ,config.OPENAI_API_KEY, interview_guide)
content_success= analyzer.content_analysis()
if not content_success:
    print("Content analysis failed exitting...")
    exit()

analyzer.combine_wav_files()
analyzer.combine_mp4_files()
analyzer.performance_analysis()
analyzer.finish_analysis()'''





