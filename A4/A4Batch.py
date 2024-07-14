import argparse

import cupy as cp
import numpy as np

from utils.utils import read_wave_file
from time import time


def sliding_window_fft_batch(data, window_size, offset, batch_size):
    data = cp.asarray(data, dtype=cp.float64)
    n_windows = (len(data) - window_size) // offset + 1
    sum_abs_fft_results = cp.zeros(window_size // 2)

    print('Batchgröße: ', batch_size)

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)

        # Berechne start und ende des aktuellen batches
        start = batch_start * offset
        end = start + (batch_end - batch_start) * offset + window_size
        end = min(end, len(data))

        # Erstellt die View auf das aktuelle Fenster
        batch_windows = cp.lib.stride_tricks.as_strided(data[start:end],
                                                        shape=(batch_end - batch_start, window_size),
                                                        strides=(offset * data.itemsize, data.itemsize))

        # FFT berechnen
        batch_fft_results = cp.fft.fft(batch_windows, axis=1)
        abs_fft_results = cp.abs(batch_fft_results)

        # Sum results across all windows in the batch
        sum_abs_fft_results += cp.sum(abs_fft_results[:, :window_size // 2], axis=0)

    avg_fft_results = sum_abs_fft_results / n_windows

    return avg_fft_results.get()


def get_frequency(amplitude_means, block_size, sample_rate, threshold):
    # Berechne Frequenzen
    freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)
    # Ausgabe der Frequenzen, die den Threshold übersteigen
    for freq, amp in zip(freq_bins[:block_size // 2], amplitude_means):
        if amp > threshold:
            print(f'Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}')


def calculate(audio_data, sample_rate, block_size, offset, threshold, batch_size=55296):
    start = time()
    sum_abs_fft_results = sliding_window_fft_batch(audio_data, block_size, offset, batch_size)
    get_frequency(sum_abs_fft_results, block_size, sample_rate, threshold)
    end = time()
    return end - start


def main():
    parser = argparse.ArgumentParser(description='Analyze WAV file for frequency components using FFT.')
    parser.add_argument('filename', type=str, help='Path to the WAV file')
    parser.add_argument('block_size', type=int, help='Block size (base 2)')
    parser.add_argument('offset', type=int, help='Offset between blocks (between 1 and block size)')
    parser.add_argument('threshold', type=float, help='Threshold for amplitude mean')
    parser.add_argument('--batch_size', type=int, default=55296, help='Batch size for FFT processing (default: 55296)')

    args = parser.parse_args()

    sample_rate, audio_data = read_wave_file(args.filename)
    duration = calculate(audio_data, sample_rate, args.block_size, args.offset, args.threshold, args.batch_size)

    print('Sekunden: ', duration)


if __name__ == '__main__':
    main()
