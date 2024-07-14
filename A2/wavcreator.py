import wave
import numpy as np
import argparse


def generate_wave_file(filename, duration, sample_rate, freq_amp_pairs):
    num_samples = int(sample_rate * duration)
    wave_data = np.zeros(num_samples, dtype=np.float32)

    for freq, amp in freq_amp_pairs:
        t = np.linspace(0, duration, num_samples, endpoint=False)
        wave_data += amp * np.sin(2 * np.pi * freq * t)

    # Normalisierung der Wellenform
    max_amp = np.max(np.abs(wave_data))
    if max_amp > 0:
        wave_data /= max_amp

    # Konvertiert Welle in 16-Bit PCM-Format
    wave_data = np.int16(wave_data * 32767)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16 bits per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_data.tobytes())


def parse_freq_amp_pairs(freq_amp_str):
    pairs = freq_amp_str.split(',')
    freq_amp_pairs = []
    for pair in pairs:
        freq, amp = map(float, pair.split(':'))
        freq_amp_pairs.append((freq, amp))
    return freq_amp_pairs


def main():
    parser = argparse.ArgumentParser(description='Generate a WAV file.')
    parser.add_argument('filename', type=str, help='Output WAV file name')
    parser.add_argument('duration', type=float, help='Duration of the WAV file in seconds')
    parser.add_argument('freq_amp_pairs', type=str,
                        help='Comma-separated list of frequency:amplitude pairs (e.g., "440:0.5,880:0.3")')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the WAV file (default: 44100)')

    args = parser.parse_args()

    freq_amp_pairs = parse_freq_amp_pairs(args.freq_amp_pairs)
    generate_wave_file(args.filename, args.duration, args.sample_rate, freq_amp_pairs)


if __name__ == '__main__':
    main()
