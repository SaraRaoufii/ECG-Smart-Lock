from pathlib import Path
import wfdb
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

record_path = PROJECT_DIR / "data" / "ecgiddb" / "ecg-id-database-1.0.0" / "Person_01" / "rec_1"

record = wfdb.rdrecord(str(record_path))

print("Sampling rate:", record.fs)
print("Number of samples:", record.sig_len)
print("Signal names:", record.sig_name)
print("Signal shape:", record.p_signal.shape)

raw_ecg = record.p_signal[:, 0]

filtered_ecg = record.p_signal[:, 1]

time = [i / record.fs for i in range(record.sig_len)]

plt.figure(figsize=(15, 4))
plt.plot(time, raw_ecg)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("ECG-ID - Person 01 - Record 1 - Raw ECG")

plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 4))
plt.plot(time, filtered_ecg)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("ECG-ID - Person 01 - Record 1 - Official Filtered ECG")

plt.tight_layout()
plt.show()