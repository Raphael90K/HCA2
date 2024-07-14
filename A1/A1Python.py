import numpy as np
import argparse
from utils.utils import read_wave_file
from time import time


def analyze_frequency_blocks(audio_data, sample_rate, block_size, offset, threshold):
    num_samples = len(audio_data)
    freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)

    amplitude_sums = np.zeros(block_size // 2)
    num_blocks = 0

    start_index = 0

    # Analysiert die Blöcke sequenziell in einer Schleife.
    while start_index + block_size <= num_samples:
        end_index = start_index + block_size
        block = audio_data[start_index:end_index]
        fft_result = np.fft.fft(block)
        amplitudes = np.abs(fft_result)[:block_size // 2]
        amplitude_sums += amplitudes
        num_blocks += 1
        start_index += offset

    amplitude_means = amplitude_sums / num_blocks

    for freq, amp in zip(freq_bins[:block_size // 2], amplitude_means):
        if amp > threshold:
            print(f'Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}')


def calculate(audio_data, sample_rate, block_size, offset, threshold):
    start = time()
    analyze_frequency_blocks(audio_data, sample_rate, block_size, offset, threshold)
    end = time()
    return end - start


def main():
    parser = argparse.ArgumentParser(description='Analyze WAV file for frequency components using FFT.')
    parser.add_argument('filename', type=str, help='Path to the WAV file')
    parser.add_argument('block_size', type=int, help='Block size (between 64 and 512)')
    parser.add_argument('offset', type=int, help='Offset between blocks (between 1 and block size)')
    parser.add_argument('threshold', type=float, help='Threshold for amplitude mean')
    args = parser.parse_args()
    sample_rate, audio_data = read_wave_file(args.filename)
    duration = calculate(audio_data, sample_rate, args.block_size, args.offset, args.threshold)
    print("Sekunden: ", duration)


if __name__ == '__main__':
    main()
