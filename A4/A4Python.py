import argparse

import numpy as np
import cupy as cp

from A1.readwav import read_wave_file
from benchmark import timeit


@timeit
def analyze_frequency_blocks(audio_data, sample_rate, block_size, offset, threshold):
    num_samples = len(audio_data)
    num_blocks = (num_samples - block_size) // offset + 1

    # Transfer audio data to GPU
    d_data = cp.asarray(audio_data, dtype=np.float32)

    # Allocate GPU memory for FFT input and output
    d_windowed_data = cp.zeros((num_blocks, block_size), dtype=np.float32)

    # Calculate Hamming window on GPU
    for tid in range(num_blocks):
        start_index = tid * offset
        d_windowed_data[tid] = d_data[start_index:start_index + block_size] * (
                0.54 - 0.46 * cp.cos(2.0 * np.pi * cp.arange(block_size, dtype=np.float32) / (block_size - 1))
        )

    # Compute FFT on GPU using cuFFT-like approach
    d_fft_data = cp.fft.fft(d_windowed_data, axis=1)

    # Compute amplitude means
    d_amplitude_means = cp.abs(d_fft_data).mean(axis=0)

    # Get frequency bins
    freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)

    # Print frequencies with amplitudes above threshold
    for freq, amp in zip(freq_bins[:block_size // 2], d_amplitude_means):
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
