import os
import sys

import numpy as np
import argparse
from scipy.fft import fft
from multiprocessing import Pool, Array, Manager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.readwav import read_wave_file


class ParallelFft(Pool):
    def __init__(self, audio_data: np.ndarray, sample_rate, block_size, offset, threshold, max_workers=32):
        super().__init__(max_workers=max_workers)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.offset = offset
        self.threshold = threshold

        self.num_samples = len(self.audio_data)
        self.avg_amp = None
        self.freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)
        self.num_blocks = int((self.num_samples - self.block_size) / self.offset)

        self.sums = np.zeros(self.block_size // 2)
        self.lock = Manager().Lock()

    def process_block(self, blocknum):
        start_index = blocknum * self.offset
        end_index = start_index + self.block_size
        block = self.audio_data[start_index:end_index]
        fft_result = fft(block)
        amplitudes = np.abs(fft_result)[:self.block_size // 2]
        with self.lock:
            self.sums += amplitudes

    def analyze_frequency_blocks(self, max_workers=32):
        with self:
            self.map(self.process_block, range(self.num_blocks))

        self.avg_amp = self.sums / self.num_blocks

        for freq, amp in zip(self.freq_bins[:self.block_size // 2], self.avg_amp):
            if amp > self.threshold:
                print(f"Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze WAV file for frequency components using FFT.')
    parser.add_argument('filename', type=str, help='Path to the WAV file')
    parser.add_argument('block_size', type=int, help='Block size (between 64 and 512)')
    parser.add_argument('offset', type=int, help='Offset between blocks (between 1 and block size)')
    parser.add_argument('threshold', type=float, help='Threshold for amplitude mean')
    parser.add_argument('--max_workers', type=int, default=32, help='Maximum number of parallel workers (default: 32)')

    args = parser.parse_args()

    sample_rate, audio_data = read_wave_file(args.filename)
    parallel_fft = ParallelFft(audio_data, sample_rate, args.block_size, args.offset, args.threshold, args.max_workers)
    parallel_fft.analyze_frequency_blocks()


if __name__ == '__main__':
    main()
