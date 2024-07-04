import wave
import numpy as np
import argparse
from scipy.fft import fft


def read_wave_file(filename):
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        return sample_rate, audio_data


def analyze_frequency_blocks(audio_data, sample_rate, block_size, offset, threshold):
    num_samples = len(audio_data)
    num_blocks = (num_samples - block_size) // offset + 1
    freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)

    amplitude_sums = np.zeros(block_size // 2)

    for i in range(num_blocks):
        start_index = i * offset
        end_index = start_index + block_size
        block = audio_data[start_index:end_index]
        windowed_block = block * np.hanning(block_size)
        fft_result = fft(windowed_block)
        amplitudes = np.abs(fft_result)[:block_size // 2]
        amplitude_sums += amplitudes

    amplitude_means = amplitude_sums / num_blocks

    for freq, amp in zip(freq_bins[:block_size // 2], amplitude_means):
        if amp > threshold:
            print(f"Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze WAV file for frequency components using FFT.')
    parser.add_argument('filename', type=str, help='Path to the WAV file')
    parser.add_argument('block_size', type=int, help='Block size (between 64 and 512)')
    parser.add_argument('offset', type=int, help='Offset between blocks (between 1 and block size)')
    parser.add_argument('threshold', type=float, help='Threshold for amplitude mean')

    args = parser.parse_args()

    sample_rate, audio_data = read_wave_file(args.filename)
    analyze_frequency_blocks(audio_data, sample_rate, args.block_size, args.offset, args.threshold)


if __name__ == '__main__':
    main()
