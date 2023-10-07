import os
import tkinter as tk
from tkinter import filedialog
from pygame import mixer
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize variables
audio_data = None
fs = None
precomputed_spectrogram = None
current_time = 0
y_max_global = 22500

# Initialize pygame mixer
mixer.init()

# Initialize Tkinter
root = tk.Tk()
root.title("Audio Spectrogram")

# Initialize Tkinter variables for color mapping
color_map = tk.StringVar(value='gray_r')

# Initialize figure for plotting
fig, ax = plt.subplots()

def load_file():
    global audio_data, fs, precomputed_spectrogram
    stop_audio()
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.flac *.ogg *.mp3")])
    audio_data, fs = librosa.load(file_path, sr=44100)
    
    # Precompute the spectrogram
    f, t, Sxx = spectrogram(audio_data, fs, nperseg=2048, nfft=4096)
    precomputed_spectrogram = 10 * np.log10(Sxx + np.finfo(float).eps)
    
    # Save to temp file for playback
    temp_path = "temp.wav"  # Salvando na pasta atual para evitar problemas de permissão
    with open(temp_path, 'wb') as fid:
        wavfile.write(fid, fs, (audio_data * 32767).astype(np.int16))

    update_spectrogram(0)

def on_scroll(event):
    global y_max_global
    _, y_max = ax.get_ylim()
    zoom_factor = 1.1
    if event.delta > 0:
        y_max_global = y_max / zoom_factor  # atualize y_max_global
    else:
        y_max_global = y_max * zoom_factor  # atualize y_max_global
    ax.set_ylim([0, y_max_global])  # Atualize o limite superior do eixo y
    canvas.draw()

def play_audio():
    home = os.path.expanduser("~")
    temp_path = os.path.join(home, "temp.wav")
    mixer.music.load(temp_path)
    mixer.music.play()
    root.after(100, audio_timer)
    root.after(100, update_spectrogram_timer)

def pause_audio():
    global current_time
    current_time = mixer.music.get_pos() / 1000.0
    mixer.music.pause()

def stop_audio():
    global current_time
    current_time = 0
    mixer.music.stop()

def audio_timer():
    if mixer.music.get_busy():
        root.after(100, audio_timer)

def update_spectrogram_timer():
    if mixer.music.get_busy():
        current_time = mixer.music.get_pos() / 1000.0  # em segundos
        update_spectrogram(current_time)
        root.after(17, update_spectrogram_timer)

def update_spectrogram(start_time):
    global precomputed_spectrogram, fs, color_map, y_max_global
    ax.clear()
    if precomputed_spectrogram is None:
        return
    start_idx = int((start_time - 1.0) * precomputed_spectrogram.shape[1] / (len(audio_data) / fs))
    end_idx = int((start_time + 1.0) * precomputed_spectrogram.shape[1] / (len(audio_data) / fs))
    Sxx_segment = precomputed_spectrogram[:, max(0, start_idx):min(end_idx, precomputed_spectrogram.shape[1])]

    # Cálculo do 70º percentil para dB_min
    dB_min = np.percentile(Sxx_segment, 70)
    dB_max = np.max(Sxx_segment)

    freqs = np.linspace(0, fs // 2, precomputed_spectrogram.shape[0])
    
    im = ax.imshow(Sxx_segment, aspect='auto', cmap=color_map.get(), origin='lower',
                   extent=[0, 2, freqs.min(), freqs.max()],
                   vmin=dB_min, vmax=dB_max)  # Utilize dB_min aqui
    
    ax.axvline(x=1, color='r', linestyle='--')
    ax.set_ylim([freqs.min(), y_max_global])  # Utilize y_max_global aqui
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    canvas.draw()

# Buttons
load_button = tk.Button(root, text="Load File", command=load_file)
load_button.pack()
play_button = tk.Button(root, text="Play", command=play_audio)
play_button.pack()
pause_button = tk.Button(root, text="Pause", command=pause_audio)
pause_button.pack()
stop_button = tk.Button(root, text="Stop", command=stop_audio)
stop_button.pack()

# Dropdown menu for color maps
color_maps = ['gray', 'gray_r', 'plasma', 'inferno', 'magma', 'cividis']
drop_menu = tk.OptionMenu(root, color_map, *color_maps)
drop_menu.pack()

# Spectrogram Canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
canvas.get_tk_widget().bind("<MouseWheel>", on_scroll)

root.mainloop()
