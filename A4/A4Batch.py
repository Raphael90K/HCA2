import argparse
import os
import sys

import cupy as cp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from readwav import read_wave_file
from benchmark import timeit

@timeit
def sliding_window_fft_batch(data, window_size, offset, batch_size):
    data = cp.asarray(data, dtype=cp.float64)
    n_windows = (len(data) - window_size) // offset + 1
    sum_abs_fft_results = cp.zeros(window_size // 2)

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)

        # Calculate start and end indices for the current batch
        start = batch_start * offset
        end = start + (batch_end - batch_start) * offset + window_size

        # Ensure end doesn't go beyond the data length
        end = min(end, len(data))

        # Create strided view of the data for the current batch
        batch_windows = cp.lib.stride_tricks.as_strided(data[start:end],
                                                        shape=(batch_end - batch_start, window_size),
                                                        strides=(offset * data.itemsize, data.itemsize))

        # Perform FFT on the windowed data
        batch_fft_results = cp.fft.fft(batch_windows, axis=1)
        abs_fft_results = cp.abs(batch_fft_results)

        # Sum results across all windows in the batch
        sum_abs_fft_results += cp.sum(abs_fft_results[:, :window_size // 2], axis=0)

    avg_fft_results = sum_abs_fft_results / n_windows

    return avg_fft_results.get()


def get_frequency(amplitude_means, block_size, sample_rate, threshold):
    # Get frequency bins
    freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)
    # Print frequencies with amplitudes above threshold
    for freq, amp in zip(freq_bins[:block_size // 2], amplitude_means):
        if amp > threshold:
            print(f"Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze WAV file for frequency components using FFT.')
    parser.add_argument('filename', type=str, help='Path to the WAV file')
    parser.add_argument('block_size', type=int, help='Block size (between 64 and 512)')
    parser.add_argument('offset', type=int, help='Offset between blocks (between 1 and block size)')
    parser.add_argument('threshold', type=float, help='Threshold for amplitude mean')
    parser.add_argument('--batch_size', type=int, default=55296, help='Batch size for FFT processing (default: 55296)')

    args = parser.parse_args()

    sample_rate, audio_data = read_wave_file(args.filename)
    sum_abs_fft_results = sliding_window_fft_batch(audio_data, args.block_size, args.offset, args.batch_size)
    get_frequency(sum_abs_fft_results, args.block_size, sample_rate, args.threshold)


if __name__ == '__main__':
    main()
