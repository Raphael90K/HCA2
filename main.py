import cupy as cp
import cupyx.scipy.fft as cufft

# Funktion zur Berechnung der maximalen Batchgröße basierend auf dem verfügbaren Speicher
def calculate_max_batch_size(window_size, dtype=cp.float32):
    total_memory = cp.cuda.runtime.memGetInfo()[1]  # Maximal verfügbarer Speicher in Bytes
    element_size = cp.dtype(dtype).itemsize  # Größe eines Elements im angegebenen Datentyp in Bytes

    max_batch_size = int((total_memory * 0.8) / (window_size * element_size))  # 80% des verfügbaren Speichers verwenden

    return max_batch_size

# Generiere zufällige Daten auf der GPU
data_size = 5200000  # 512 MB Daten
data_gpu = cp.random.randn(data_size).astype(cp.float32)

# Parameters for the FFT
window_size = 256
offset = 1

# Berechne die maximale Batchgröße basierend auf dem verfügbaren Speicher
max_batch_size = calculate_max_batch_size(window_size)

print(f"Maximal mögliche Batchgröße: {max_batch_size}")

# Initialisiere Ergebnis-Arrays auf der GPU
fft_results_gpu = cp.empty((max_batch_size, window_size), dtype=cp.complex64)
abs_sum_gpu = cp.zeros(window_size, dtype=cp.float32)

# Führe batchweise FFTs aus und überwache die Speicherauslastung
total_memory = cp.cuda.runtime.memGetInfo()[1] * 0.6 # Maximal verfügbarer Speicher in Bytes
batch_size = max_batch_size

for i in range(0, data_size - window_size + 1, batch_size * offset):
    # Bestimme die tatsächliche Batchgröße für diese Iteration
    current_batch_size = min(batch_size, (data_size - window_size - i) // offset + 1)

    # Sammle Fensterdaten für die aktuelle Batchgröße
    batch_data = []
    for j in range(current_batch_size):
        start_idx = i + j * offset
        end_idx = start_idx + window_size
        batch_data.append(data_gpu[start_idx:end_idx])

    # Konvertiere die Liste in ein CuPy-Array
    batch_data_gpu = cp.stack(batch_data)

    # Führe batchweise FFTs auf `batch_data_gpu` durch
    batched_fft_results = cufft.fft(batch_data_gpu)

    # Berechne absolute Werte und summiere sie auf
    abs_batched_fft_results = cp.abs(batched_fft_results)
    abs_sum_gpu += cp.sum(abs_batched_fft_results, axis=0)

    # Überwache die Speicherauslastung während der Ausführung
    used_memory = cp.cuda.memory.MemoryPool().used_bytes()
    free_memory = total_memory - used_memory
    print(f"Iteration {i}: Used GPU Memory: {used_memory / (1024**3)} GB, Free GPU Memory: {free_memory / (1024**3)} GB")

# Drucke oder verwende die Summe der absoluten Werte, wie benötigt
print("Summe der FFT-Ergebnisse:", abs_sum_gpu.get())
