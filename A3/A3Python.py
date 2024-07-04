import wave
import numpy as np
import argparse
from scipy.fft import fft
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_wave_file(filename):
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


def process_block(audio_data, start_index, block_size):
    end_index = start_index + block_size
    block = audio_data[start_index:end_index]
    windowed_block = block * np.hanning(block_size)
    fft_result = fft(windowed_block)
    amplitudes = np.abs(fft_result)[:block_size // 2]
    return amplitudes


def analyze_frequency_blocks(audio_data, sample_rate, block_size, offset, threshold, max_workers=4):
    num_samples = len(audio_data)
    freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)

    amplitude_sums = np.zeros(block_size // 2)
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        start_index = 0
        while start_index + block_size <= num_samples:
            futures.append(executor.submit(process_block, audio_data, start_index, block_size))
            start_index += offset

        num_blocks = 0
        for future in as_completed(futures):
            amplitudes = future.result()
            amplitude_sums += amplitudes
            num_blocks += 1

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
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of parallel workers (default: 4)')

    args = parser.parse_args()

    sample_rate, audio_data = read_wave_file(args.filename)
    analyze_frequency_blocks(audio_data, sample_rate, args.block_size, args.offset, args.threshold, args.max_workers)


if __name__ == '__main__':
    main()
