import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, irfft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Function to generate and plot a spectrogram
def plot_spectrogram(audio, sample_rate, title, freq_min=None, freq_max=None, cmap='viridis'):
    """
    Generates and plots a spectrogram of the audio data.
    
    Parameters:
    audio: numpy array, The audio data.
    sample_rate: int, The sample rate of the audio data.
    title: str, The title for the plot.
    freq_min: int, optional, The minimum frequency to display.
    freq_max: int, optional, The maximum frequency to display.
    cmap: str, optional, The colormap for the spectrogram.
    """
    f, t, Sxx = spectrogram(audio, sample_rate)
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    if freq_min is not None and freq_max is not None:
        plt.ylim(freq_min, freq_max)
    plt.colorbar(label='Intensity [dB]')
    plt.show()

def read_wav_file(filepath):
    # Read the existing WAV file
    sample_rate, signal = wavfile.read(filepath)
    signal = signal.astype(np.float32)

    # Ensure signal is in the correct format (mono)
    if len(signal.shape) > 1:
        signal = signal[:, 0] # Take the first channel if stereo

    return signal

def draw_pixel(signal, sample_rate, x_seconds, y_freq): # freq is kinda equivalent to y position of the pixel

    duration = len(signal) / sample_rate
    t = np.linspace(0, duration, len(signal), endpoint=False)

    # Define the time interval for the pixel
    pixel_duration = 1  # Duration of the pixel in seconds, adjust as needed
    pixel_start = x_seconds - 0.5 * pixel_duration
    pixel_end = x_seconds + 0.5 * pixel_duration

    # Generate the pixel
    pixel_indices = np.logical_and(t >= pixel_start, t <= pixel_end)
    pixel_intensity = 100000  # Adjust this value as needed for visibility
    pixel_signal = pixel_intensity * np.sin(2 * np.pi * y_freq * t[pixel_indices])

    # Overlay the pixel signal onto the original signal
    new_signal = np.copy(signal)
    new_signal[pixel_indices] += pixel_signal

    # Normalize the signal
    #new_signal = new_signal / np.max(np.abs(new_signal))
    return new_signal

def main():
    sample_rate = 44100  # Sample rate in Hz
    filepath = "./audio_clips/popular_songs/Britney_Spears_Toxic.wav"
    
    # read the WAV file once to get the signal, which we will add pixels too. 
    # read it once at the start so we can append pixels to the file (call draw_pixel multiple times)
    read_signal = read_wav_file(filepath)

    # Draws a pixel (more like a line shape) at a 'x' and 'y' coordinate on the spectogram.
    # X is the time (given in seconds)
    # y is the frequency

    # Example: Draws a single pixel at x:5 seconds and y:20000 hz frequency
    read_signal = draw_pixel(read_signal,sample_rate, 5,20000) # x and y, 5 seconds at 20000 hz.

    # Draw a bunch of pixels going across the top at 20000 hz
    read_signal = draw_pixel(read_signal,sample_rate, 5,20000)
    read_signal = draw_pixel(read_signal,sample_rate, 10,20000)
    read_signal = draw_pixel(read_signal,sample_rate, 15,20000)
    read_signal = draw_pixel(read_signal,sample_rate, 20,20000)

    # # Draw pixels at a different y pos
    read_signal = draw_pixel(read_signal,sample_rate, 8,15000)
    read_signal = draw_pixel(read_signal,sample_rate, 13,16000)
    read_signal = draw_pixel(read_signal,sample_rate, 25,10000)
    final = draw_pixel(read_signal,sample_rate, 35,21000)

    # now that we can control the x and y pos we could basically create letters
    
    # This gives it the signal "array" of audio data, not the file itself. File itself not created/modified yet.
    plot_spectrogram(final, sample_rate, 'Toxic', cmap='inferno')

if __name__ == "__main__":
    main()