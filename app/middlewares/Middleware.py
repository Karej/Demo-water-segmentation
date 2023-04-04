# required libraries
import re
import pandas as pd
import numpy as np
import librosa
import csv
import soundfile as sf
from tqdm import tqdm
from tqdm.auto import tqdm
import os
from functools import wraps
from app.model.Access_data import Access_data
from app.model.User import User
from flask_jwt_extended import get_jwt_identity

tqdm.pandas()


# access data checker
def data_access_decorator(f):
    @wraps(f)
    def _data_access_decorator(*args, **kwargs):
        # before execution
        # handle version, user
        version = kwargs['version']
        identity = get_jwt_identity()
        
        user = User.query.filter_by(id=identity).first()
        isAdmin = True if user.accountRole == 'admin' else False

        if not isAdmin:
            access_data = Access_data.query.filter_by(user_id=identity, data_version=version).first()
            if not access_data:
                return {"msg": "not accessible"}, 400

        # execution
        result = f(*args, **kwargs)
        
        #after execution 

        return result
    return _data_access_decorator


# for create training file for TTS
def tts_upload_decorator(f):
    @wraps(f)
    def _tts_upload_decorator(*args, **kwargs):
        # before execution

        # execution
        result = f(*args, **kwargs)
        
        #after execution
        
        if result['type'] == 'tts': 
            parent_dir =  os.path.join(os.getcwd(), 'uploads', 'tts', result['filename'])

            # THIS PART IS FOR TRAINING DATA

            # input csv file containing text
            descripted_filepath = os.path.join(parent_dir, 'test', 'transcript.csv')
            # input audio file

            descripted_file = pd.read_csv(f"{descripted_filepath}")

            # THIS PART IS FOR TESTING DATA
            # pass any null values
            descripted_file = descripted_file[descripted_file['text'].notna()]
            descripted_file = descripted_file[descripted_file['filepath'].notna()]

            # measuring time series of audio files
            def process(path):
                absolute_path = os.path.join(parent_dir, path)
                waveform, sample_rate = librosa.load(absolute_path) # audio time series
                print(absolute_path)
                if waveform.ndim > 1:
                    waveform = waveform[:, 0]           
                return waveform.shape[0]/sample_rate
            
            # return absolute filepath
            def absolute_filepath(path):
                return os.path.join(parent_dir, path)

            descripted_file['length'] =  descripted_file['filepath'].progress_apply(process)
            descripted_file['filepath'] = descripted_file['filepath'].progress_apply(absolute_filepath)

            training_set = descripted_file # later for training data
            training_set['text'] = training_set['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())  # create lowercase text of files

            # write data into
            for file_path in tqdm(training_set['filepath']):
                y, sr = librosa.load(os.path.join(parent_dir, file_path), sr=22050, mono=True)
                sf.write(os.path.join(parent_dir, file_path), y, sr)

            required_csv_path = os.path.join(parent_dir, 'transcriptWithLength.csv')
            training_set.to_csv(required_csv_path, index=False)

        return result
    return _tts_upload_decorator
