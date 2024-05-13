#!/usr/bin/env python
# coding: utf-8

import whisperx
import gc
import os
import json
from dataclasses import dataclass, field
from typing import List
import json
from nltk.stem import WordNetLemmatizer
import wave
from pydub import AudioSegment
import webrtcvad
import parselmouth
from parselmouth.praat import call
import statistics
import librosa
import numpy as np
from fer import FER
from fer import Video
import pandas as pd
import csv
import shutil
import json
import re
import json
import os
from glob import glob
import numpy as np
import os
import pandas as pd
from joblib import load
import cv2


# ## Jobtalk Analyzer

def calculate_column_averages(csv_file):
    # Dictionary to store column sums and counts
    column_sums = {}
    column_counts = {}

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header row

        # Initialize sums and counts for each column
        for header in headers:
            if header.endswith('0') and header != 'box0':
                column_sums[header] = 0
                column_counts[header] = 0

        # Iterate over rows for each column
        line_count = 0
        for row in reader:
            line_count += 1
            for column_index, value in enumerate(row):
                header = headers[column_index]
                if header.endswith('0') and header != 'box0':
                    if value:
                        column_sums[header] += float(value)
                        column_counts[header] += 1

        print("Total lines read:", line_count)

    # Calculate averages
    column_averages = {}
    for header, total in column_sums.items():
        count = column_counts[header]
        column_averages[header] = total / count if count != 0 else 0

    return column_averages


def convert_csv_json(csv_file):
# Extract the base name for the CSV file to use for the JSON file name
    base_name = os.path.basename(csv_file).replace('.csv', '.json')

    # Initialize a dictionary to hold the data
    data = {}

    # Read the CSV file and convert to dictionary
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # The first column is assumed to be 'Column' and the second 'Average'
            column_name = row['Column']
            average_value = row['Average']
            # Convert the average value to a float
            average_value = float(average_value)
            data[column_name] = average_value

    # Write to JSON file
    #Write to the same csv file but change .csv to .json
    json_file_path = csv_file.replace('.csv', '.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Assuming the categories dictionary and phrases are already defined
phrases = {
    "i don't know": 'Non-fluencies',
    "i mean": 'Non-fluencies',
    "oh well": 'Non-fluencies',
    "you know": 'Non-fluencies',
    "a lot": 'Quantifiers'
}
def read_categories(csv_file):
    categories = {}
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            category = row[0]
            words = row[1:]
            categories[category] = words
    return categories

# Function to check if a word matches the word or stem in the category
def match_word_in_category(word, category_words):
    for category_word in category_words:
        if category_word.endswith('*'):
            stem = category_word[:-1]
            if word.lower().startswith(stem.lower()):
                return True
        elif word.lower() == category_word.lower():
            return True
    return False

# Function to count occurrences of words from each category in the given text
def count_words_in_transcript(transcript, categories):
    word_counts = {category: 0 for category in categories}
    words = [entry['word'].strip(".,?!").lower() for entry in transcript]  # Cleaning and extracting words

    for word in words:
        for category, category_words in categories.items():
            if match_word_in_category(word, category_words):
                word_counts[category] += 1

    text = " ".join(words)
    for phrase, category in phrases.items():
        count = text.count(phrase)
        word_counts[category] += count

    return word_counts

# Function to analyze a whisperx transcript
def analyze_transcript(transcript_json, categories):
    # Load transcript data if it's a JSON string, otherwise assume it's already a Python object
    if isinstance(transcript_json, str):
        transcript = json.loads(transcript_json)
    else:
        transcript = transcript_json

    # Count the words based on the categories
    word_counts = count_words_in_transcript(transcript, categories)
    return word_counts


def normalize_feature(feature_file, stats_file_path, output_file):
    # Load statistics
    with open(stats_file_path, 'r') as file:
        feature_stats = json.load(file)

    # Load the feature data
    with open(feature_file, 'r') as file:
        data = json.load(file)

    # Normalize features
    normalized_data = {}
    for key, value in data.items():
        if key in feature_stats and feature_stats[key]['std'] != 0:  # Prevent division by zero
            normalized_data[key] = (float(value) - feature_stats[key]['mean']) / feature_stats[key]['std']
        else:
            normalized_data[key] = float(value)  # Use original value if std is zero or value conversion needed

    # Write the normalized data to a file
    with open(output_file, 'w') as file:
        json.dump(normalized_data, file, indent=4)


def load_features_from_json(json_path):
    """
    Load features from a JSON file into a pandas DataFrame.

    Args:
    json_path (str): Path to the JSON file containing feature data.

    Returns:
    pd.DataFrame: DataFrame containing the features in a single row format.
    """
    try:
        # Load JSON data as a Series to handle single line of feature values, then convert to DataFrame
        feature_series = pd.read_json(json_path, typ='series')
        features_df = pd.DataFrame([feature_series])
        return features_df
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def load_and_predict(models_directory, json_path):
    """
    Load all SVR models from a specified directory and predict the output for given features loaded from a JSON file.

    Args:
    models_directory (str): The directory where the SVR models are stored.
    json_path (str): Path to the JSON file containing the features.

    Returns:
    dict: A dictionary containing the predictions for each attribute.
    """
    # Load features from JSON
    features = load_features_from_json(json_path)
    if features is None:
        return None

    predictions = {}
    # Check if the provided directory exists
    if not os.path.exists(models_directory):
        print(f"Error: The directory '{models_directory}' does not exist.")
        return None

    # Load each model and perform predictions
    for model_filename in os.listdir(models_directory):
        if model_filename.endswith('_model.joblib'):
            attr_name = model_filename.replace('_model.joblib', '')
            model_path = os.path.join(models_directory, model_filename)
            model = load(model_path)
            # Predict and store results using the loaded model
            predictions[attr_name] = model.predict(features)

    return predictions

@dataclass
class TranscriptionElement:
    start: float
    end: float

    def duration(self) -> float:
        return self.end - self.start

@dataclass
class Pause(TranscriptionElement):
    pause_elements: list[TranscriptionElement]

@dataclass
class FilledPause(TranscriptionElement):
    pass

@dataclass
class Silence(TranscriptionElement):
    pass

@dataclass
class Word:
    word: str
    start: float
    end: float

    def syllables(self):
        # Implement syllable counting if necessary
        pass

@dataclass
class Chunk(TranscriptionElement):
    words: List[Word]
@dataclass
class AudioAnalysis:
    transcription_elements: List[TranscriptionElement]
    total_duration: float = 0.0
    total_words: int = 0
    unique_words: set = field(default_factory=set)
    asr_score: float= 0

    def average_chunk_length_in_words(self):
        total_words_in_chunks = self.total_words
        number_of_chunks = sum(1 for element in self.transcription_elements if isinstance(element, Chunk))
        return total_words_in_chunks / number_of_chunks if number_of_chunks else 0
    def articulation_rate(self):
        total_speech_duration = sum(chunk.duration() for chunk in self.transcription_elements if isinstance(chunk, Chunk))
        return self.total_words / total_speech_duration if total_speech_duration else 0
    def mean_deviation_of_chunks_in_words(self):
        mean_chunk_length = self.average_chunk_length_in_words()
        deviations = [abs(len(chunk.words) - mean_chunk_length) for chunk in self.transcription_elements if isinstance(chunk, Chunk)]
        return sum(deviations) / len(deviations) if deviations else 0
    def duration_of_silences_per_word(self):
        total_silence_duration = sum(silence.duration() for silence in self.transcription_elements if isinstance(silence, Pause))
        return total_silence_duration / self.total_words if self.total_words else 0
    def mean_of_silence_duration(self):
        silence_durations = [silence.duration() for silence in self.transcription_elements if isinstance(silence, Pause)]
        return sum(silence_durations) / len(silence_durations) if silence_durations else 0
    def mean_duration_of_long_pauses(self):
        long_pauses = [silence.duration() for silence in self.transcription_elements if isinstance(silence, Pause) and silence.duration() >= 0.5]
        return sum(long_pauses) / len(long_pauses) if long_pauses else 0
    def frequency_of_longer_pauses_divided_by_number_of_words(self):
        long_pause_count = sum(1 for silence in self.transcription_elements if isinstance(silence, Pause) and silence.duration() >= 0.5)
        return long_pause_count / self.total_words if self.total_words else 0
    def types_divided_by_uttsegdur(self):
        #FIXME total duration should be duration of entire transcribed segment but without inter-utterance pauses
        return len(self.unique_words) / self.total_duration if self.total_duration else 0

    def mean_length_of_filled_pauses(self):
        filled_pauses = [pause.duration() for element in self.transcription_elements if isinstance(element, Pause) for pause in element.pause_elements if isinstance(pause, FilledPause)]
        return sum(filled_pauses) / len(filled_pauses) if filled_pauses else 0

    def frequency_of_filled_pauses(self):
        total_filled_pauses = sum(1 for element in self.transcription_elements if isinstance(element, Pause) for pause in element.pause_elements if isinstance(pause, FilledPause))
        return total_filled_pauses / self.total_duration if self.total_duration else 0



    def __init__ (self,word_segments, recording:str= None):
        self.recording_path= recording
        elements = []
        current_chunk_words = []
        unique= set()
        c_score= 0
        #lemmatizer = WordNetLemmatizer()

        for i, word_info in enumerate(word_segments):
            if word_info.get('start') is None:
                continue
            c_score += word_info['score']

            word = Word(word=word_info['word'], start=word_info['start'], end=word_info['end'])
            self.total_words += 1
            #TODO leematize etmek kokunu almak ama hiçbirşeye yaramadı... unique words aynı çıkıyor
            #unique.add(lemmatizer.lemmatize(word.word.lower()) )
            unique.add(word.word.lower())

            if not current_chunk_words:  # Start of a new chunk
                current_chunk_words.append(word)
            else:
                silence_duration = word.start - current_chunk_words[-1].end
                if silence_duration < 0.15:
                    current_chunk_words.append(word)
                else:
                    # End the current chunk and start a new one
                    chunk = Chunk(start=current_chunk_words[0].start, end=current_chunk_words[-1].end, words=current_chunk_words)
                    elements.append(chunk)
                    #elements.append(Pause(start=current_chunk_words[-1].end, end=word.start))
                    pauses= detect_pause_segments(current_chunk_words[-1].end*1000, word.start*1000, recording)
                    pauses_filtered= [pause for pause in pauses if pause.duration() > 0.25]
                    pause= Pause(current_chunk_words[-1].end, word.start, pauses_filtered)
                    elements.append(pause)
                    current_chunk_words = [word]

        # Add the last chunk if there are any words left
        if current_chunk_words:
            chunk = Chunk(start=current_chunk_words[0].start, end=current_chunk_words[-1].end, words=current_chunk_words)
            elements.append(chunk)

        self.transcription_elements= elements
        self.total_duration = elements[-1].end if elements else 0.0  # Duration based on the last element
        self.unique_words= unique
        self.asr_score= c_score / self.total_words


    def __str__(self):
        transcription_str = ""
        transcription_str += f"\nTotal Duration: {self.total_duration} seconds\n"
        transcription_str += f"Total Number of Words: {self.total_words}\n"
        transcription_str += f"Unique Words: {len(self.unique_words)}\n"
        transcription_str += f"Average Chunk Length in Words: {self.average_chunk_length_in_words()}\n"
        transcription_str += f"Articulation Rate: {self.articulation_rate()} words per second\n"
        transcription_str += f"Mean Deviation of Chunks in Words: {self.mean_deviation_of_chunks_in_words()}\n"
        transcription_str += f"Duration of Silences per Word: {self.duration_of_silences_per_word()} seconds/word\n"
        transcription_str += f"Mean of Silence Duration: {self.mean_of_silence_duration()} seconds\n"
        transcription_str += f"Mean Duration of Long Pauses: {self.mean_duration_of_long_pauses()} seconds\n"
        transcription_str += f"Frequency of Longer Pauses Divided by Number of Words: {self.frequency_of_longer_pauses_divided_by_number_of_words()} pauses/word\n"
        transcription_str += f"Types Divided by Uttsegdur: {self.types_divided_by_uttsegdur()} types/second\n"

        for element in self.transcription_elements:
            if isinstance(element, Pause):
                    transcription_str += f"Pause from {element.start} to {element.end} seconds\n"
                    for pause in element.pause_elements:
                        if isinstance(pause, FilledPause):
                            transcription_str += f"\tFilled Pause from {pause.start} to {pause.end} seconds\n"
            elif isinstance(element, Chunk):
                words_str = ', '.join([word.word for word in element.words])
                transcription_str += f"Chunk from {element.start} to {element.end} seconds: {words_str}\n"
        return transcription_str


    def save_features(self, file_path):
        features = self.get_features()

        with open(file_path, 'w') as file:
            json.dump(features, file, indent=4)

    def get_features(self):
        features = {
            "total_duration": self.total_duration,
            "total_words": self.total_words,
            "unique_words_count": len(self.unique_words),
            "average_chunk_length_in_words": self.average_chunk_length_in_words(),
            "articulation_rate": self.articulation_rate(),
            "mean_deviation_of_chunks_in_words": self.mean_deviation_of_chunks_in_words(),
            "duration_of_silences_per_word": self.duration_of_silences_per_word(),
            "mean_of_silence_duration": self.mean_of_silence_duration(),
            "mean_duration_of_long_pauses": self.mean_duration_of_long_pauses(),
            "frequency_of_longer_pauses_divided_by_number_of_words": self.frequency_of_longer_pauses_divided_by_number_of_words(),
            "types_divided_by_uttsegdur": self.types_divided_by_uttsegdur(),
            "mean_length_of_filled_pauses": self.mean_length_of_filled_pauses(),
            "frequency_of_filled_pauses": self.frequency_of_filled_pauses(),
            "asr_score": self.asr_score
        }

        return features

def detect_pause_segments(start, end, recording_path):
    # Load the recording using pydub
    audio = AudioSegment.from_file(recording_path)

    # Initialize the VAD
    vad = webrtcvad.Vad(3)  # 1 is the aggressiveness level

    # Extract the pause segment
    pause_audio = audio[start:end]

    # Convert to a format suitable for VAD (16kHz mono 16-bit)
    pause_audio = pause_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    pause_bytes = pause_audio.raw_data

    # Split into 30ms frames
    frame_duration = 30  # in ms
    frame_size = int(0.03 * 16000) * 2  # 30 ms * 16000 Hz * 2 bytes/sample
    frames = [pause_bytes[i:i + frame_size] for i in range(0, len(pause_bytes) - frame_size + 1, frame_size)]


    pauses = []
    current_start = start
    current_frame_length= 0
    is_current_filled = vad.is_speech(frames[0], 16000) if frames else False


    for i, frame in enumerate(frames):
        is_filled = vad.is_speech(frame, 16000)
        current_frame_length += 1
        if is_filled != is_current_filled or i == len(frames) - 1:
            # End of a segment (silence or filled), create a Pause object
            current_end = current_start + frame_duration * current_frame_length# frame length in ms
            if is_current_filled:
                pauses.append(FilledPause(start=current_start / 1000.0, end=current_end / 1000.0))
            else:
                pauses.append(Silence(start=current_start / 1000.0, end=current_end / 1000.0))
            current_start = current_end
            current_frame_length= 0
            is_current_filled = is_filled

    return pauses



class JobTalkAnalyzer:
    def __init__(self, participant_id, audio_path, video_path, candidate_id, interview_id):
        self.device = "cuda"
        self.batch_size = 16  # Adjust based on your GPU memory
        self.compute_type = "float16"  # Use "int8" for lower memory usage, with potential accuracy trade-off
        self.transcription_model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)
        self.participant_id = participant_id
        self.audio_path = audio_path
        base_path= f"analysis/{candidate_id}/{interview_id}"
        self.data_directory = os.path.join(base_path, self.participant_id)
        self.audio = whisperx.load_audio(self.audio_path)
        self.video_path = video_path

    def analyze(self):
        #self.createTranscript()
        self.prosody_analysis()
        self.facial_analysis()
        self.average_facial_features()
        self.lexical_analysis()
        self.combine_features()

    def prosody_analysis(self):
        with open(self.data_directory + "/transcription.json" , 'r') as file:
            data = json.load(file)
        self.prosody_analyzer = AudioAnalysis(data,self.audio_path )

        self.prosody_analyzer.save_features(self.data_directory + "/prosody_features.json")

    def lexical_analysis(self):
        with open(self.data_directory + "/transcription.json" , 'r') as file:
            data = json.load(file)

        categories= read_categories("liwc2007.csv")
        lex= analyze_transcript(data, categories)

        #write to file
        with open(self.data_directory + "/lexical_features.json" , 'w') as file:
            json.dump(lex, file)

    def facial_analysis(self, frequency= 8):
        face_detector = FER(mtcnn=True)
        processed_video = Video(video_file=self.video_path)

        # Analyze video and save results as CSV
        processing_data = processed_video.analyze(face_detector, display=False, frequency= frequency)
        out_dir= self.data_directory + "/facial_features.csv"
        processed_video.to_csv(processing_data, "p22.csv")
        shutil.copyfile("data.csv", out_dir)

    def combine_features(self):
        #read all features lexical_features.json, averaged_facial_features.json, prosody_features.json and combine them into a single json and write
        with open(self.data_directory + "/lexical_features.json" , 'r') as file:
            lexical_features = json.load(file)

        with open(self.data_directory + "/prosody_features.json" , 'r') as file:
            prosody_features = json.load(file)

        with open(self.data_directory + "/averaged_facial_features.json" , 'r') as file:
            facial_features = json.load(file)



        #combine all
        combined_features = {**facial_features,**lexical_features, **prosody_features}
        #write to file
        with open(self.data_directory + "/combined_features.json" , 'w') as file:
            json.dump(combined_features, file)


        feature_file= self.data_directory + "/combined_features.json"
        stats_file_path= "Oelp/scaler.json"
        output_file= self.data_directory + "/normalized_features.json"
        normalize_feature(feature_file, stats_file_path, output_file)

    def predict_scores(self):
        print(self.data_directory)
        json_path = self.data_directory + "/normalized_features.json"
        model_directory = 'Oelp'
        results = load_and_predict(model_directory, json_path)

        # Prepare a dictionary to hold the data in a structured format
        data_to_save = {}

        for attr, predictions in results.items():
            # Store the results in the dictionary instead of printing
            data_to_save[attr] = predictions[0]

        # Define the path for the JSON output file
        output_json_path = self.data_directory + "/performance_analysis.json"
        
        # Write the results to a JSON file
        with open(output_json_path, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)

    def average_facial_features(self):
        output_file = self.data_directory + "/averaged_facial_features.csv"
        column_averages = calculate_column_averages(self.data_directory + "/facial_features.csv")

        # Write averages to new CSV file
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Column', 'Average'])  # Write header row
            for column, average in column_averages.items():
                writer.writerow([column, average])

        convert_csv_json(output_file)

    def createTranscript(self):

        os.makedirs(self.data_directory, exist_ok=True)
        transcription_result = self.transcription_model.transcribe(self.audio, batch_size=self.batch_size)

        # Load alignment model based on detected language
        align_model, metadata = whisperx.load_align_model(language_code=transcription_result["language"], device=self.device)

        # Perform alignment
        aligned_result = whisperx.align(transcription_result["segments"], align_model, metadata, self.audio, self.device, return_char_alignments=False)
        print("sa")
        # Define the path for the output JSON file
        output_path = self.data_directory+ "/transcription.json"

        # Save the aligned transcription result to a JSON file
        with open(output_path, "w") as outfile:
            json.dump(aligned_result["word_segments"], outfile)




if __name__ == "__main__":
    analyzer= JobTalkAnalyzer("bad", "bad.wav","bad.mp4")
    analyzer.analyze()
    analyzer.predict_scores()
