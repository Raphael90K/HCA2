import numpy as np
import argparse
from multiprocessing import Process, Manager, cpu_count
from utils.utils import read_wave_file

from time import time


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

        self.sums = np.zeros(self.block_size // 2)

    def worker_task(self, start_index, step, result_queue):
        local_sum = np.zeros(self.block_size // 2)
        blocks = 0
        for i in range(start_index, len(self.audio_data) - self.block_size, step):
            if i + self.block_size >= len(self.audio_data):
                break
            segment = self.audio_data[i:i + self.block_size]
            fft_result = np.fft.fft(segment)
            local_sum += np.abs(fft_result)[:len(segment) // 2]
            blocks += 1
        result_queue.append((local_sum, blocks))

    def analyze_frequency_blocks(self):
        result_list = Manager().list()
        # Startindex für jeden Worker
        num_cores = min(self.max_workers, cpu_count())

        # Erstelle und starte die Worker
        processes = []
        for start in range(num_cores):
            p = Process(target=self.worker_task, args=(start, num_cores * self.offset, result_list))
            processes.append(p)
            p.start()

        # Warte darauf, dass alle Worker fertig sind
        for p in processes:
            p.join()

        # Sammle die Ergebnisse von der Queue
        num_blocks = 0
        print(result_list)
        for res in result_list:
            self.sums += res[0]
            num_blocks += res[1]

        # Berechnung des Durchschnitts der Amplituden
        self.avg_amp = self.sums / num_blocks

        # Ausgabe der relevanten Frequenzen und ihrer Amplitudenmittelwerte
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

    start = time()
    sample_rate, audio_data = read_wave_file(args.filename)
    parallel_fft = ParallelFft(audio_data, sample_rate, args.block_size, args.offset, args.threshold, args.max_workers)
    parallel_fft.analyze_frequency_blocks()
    end = time()
    print("Sekunden: ", end - start)


if __name__ == '__main__':
    main()
