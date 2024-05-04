import os
from pydub import AudioSegment
from moviepy.editor import concatenate_videoclips, VideoFileClip
import re
from JobTalkAnalyzer import JobTalkAnalyzer

class Analyzer:
    def __init__(self, path: str, pid: int):
        self.path = path
        self.wav_files = []
        self.mp4_files = []
        self._validate_files()
        self.pid= pid

    def _validate_files(self):
        # List all files in the directory
        files = os.listdir(self.path)
        # Filter out .wav and .mp4 files
        self.wav_files = [f for f in files if f.endswith('.wav')]
        self.mp4_files = [f for f in files if f.endswith('.mp4')]

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
        analyzer= JobTalkAnalyzer(self.pid, audio_path, video_path)
        analyzer.analyze()
        analyzer.predict_scores()


# Example usage
analyzer = Analyzer('content')
analyzer.combine_wav_files()
analyzer.combine_mp4_files()
