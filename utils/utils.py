import wave

import numpy as np


def read_wave_file(filename):
    '''
    Einlesen der .wav von filename
    return: (sample_rate, audio_data)
    '''
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        audio_data = wav_file.readframes(num_frames)

        if num_channels == 1:
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_data = np.frombuffer(audio_data, dtype=np.int16).reshape(-1, num_channels)
            audio_data = audio_data[:, 0]

        return sample_rate, audio_data

