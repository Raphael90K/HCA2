import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from utils.readwav import read_wave_file  # Annahme: Diese Funktion liest die WAV-Datei ein


class ParallelFft:
    def __init__(self, audio_data: np.ndarray, sample_rate, block_size, offset, threshold, max_workers=32):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.offset = offset
        self.threshold = threshold
        self.max_workers = max_workers

        self.num_samples = len(self.audio_data)
        self.avg_amp = None
        self.freq_bins = np.fft.fftfreq(block_size, 1 / sample_rate)
        self.num_blocks = int((self.num_samples - self.block_size) / self.offset)

        self.sums = np.zeros(self.block_size // 2)

    def process_block(self, blocknum):
        start_index = blocknum * self.offset
        end_index = start_index + self.block_size
        block = self.audio_data[start_index:end_index]
        fft_result = np.fft.fft(block)
        amplitudes = np.abs(fft_result)[:self.block_size // 2]

        return amplitudes

    def process_result(self, amp):
        print(amp)

    def analyze_frequency_blocks(self):
        print(f"Total number of blocks: {self.num_blocks}")

        # Pool von Prozessen erstellen (Anzahl der Kerne verwenden)
        num_cores = min(self.max_workers, cpu_count())
        print(num_cores)

        with Pool(processes=num_cores) as pool:
            result = pool.map_async(self.process_block, range(self.num_blocks))

        print(result.get())
        self.sums += np.sum(result, axis=0)
        print(self.sums)

        # Berechnung des Durchschnitts der Amplituden
        avg_amp = self.sums / self.num_blocks

        # Ausgabe der relevanten Frequenzen und ihrer Amplitudenmittelwerte
        for freq, amp in zip(self.freq_bins[:self.block_size // 2], avg_amp):
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
