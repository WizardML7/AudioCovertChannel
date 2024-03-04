import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, irfft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Function to modify frequency components and blot out a specific range
def modify_frequency(audio_channel, sample_rate, freq_min, freq_max):
    """
    Zeroes out the specified frequency range in the audio data.
    
    Parameters:
    audio_channel: numpy array, The audio data for one channel.
    sample_rate: int, The sample rate of the audio data.
    freq_min: int, The minimum frequency to blot out.
    freq_max: int, The maximum frequency to blot out.
    
    Returns:
    modified_channel: numpy array, The modified audio data for one channel.
    """
    # Perform Fourier Transform
    fft_data = rfft(audio_channel)
    # Calculate the frequency bins
    freq_bins = np.fft.rfftfreq(len(audio_channel), 1/sample_rate)
    # Find indices of frequencies in the specified range and zero out
    indices = np.where((freq_bins >= freq_min) & (freq_bins <= freq_max))
    fft_data[indices] = 0
    # Perform Inverse Fourier Transform
    modified_channel = irfft(fft_data)
    
    return modified_channel

# Function to isolate a frequency range to reveal hidden messages
def isolate_frequency_range(audio_data, sample_rate, freq_min, freq_max):
    """
    Isolates the specified frequency range in the audio data to reveal hidden messages.
    
    Parameters:
    audio_data: numpy array, The audio data.
    sample_rate: int, The sample rate of the audio data.
    freq_min: int, The minimum frequency of the range to isolate.
    freq_max: int, The maximum frequency of the range to isolate.
    
    Returns:
    isolated_audio: numpy array, The audio data with the specified frequency range isolated.
    """
    # Perform Fourier Transform
    fft_data = rfft(audio_data)
    # Calculate the frequency bins
    freq_bins = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    # Zero out frequencies outside the specified range
    fft_data[(freq_bins < freq_min) | (freq_bins > freq_max)] = 0
    # Perform Inverse Fourier Transform
    isolated_audio = irfft(fft_data)
    
    return isolated_audio

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

def encode_visual_A(sample_rate, duration, freq_min, freq_max):
    """
    Generates a synthetic audio signal that visually encodes the letter 'A' in its spectrogram, with improved accuracy.
    
    Parameters:
    sample_rate: int, The sample rate of the audio data.
    duration: float, The duration of the signal in seconds.
    freq_min: int, The minimum frequency of the 'A' shape.
    freq_max: int, The maximum frequency of the 'A' shape.
    
    Returns:
    signal: numpy array, The generated audio signal.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Create a silent signal as the base
    signal = np.zeros_like(t)
    
    # Calculate frequencies for the legs and crossbar of 'A'
    freq_leg1 = freq_min  # Starting frequency for the first leg
    freq_leg2 = freq_max  # Ending frequency for the second leg
    freq_crossbar = (freq_min + freq_max) / 2  # Frequency for the crossbar
    
    # Time points to create the 'A' shape
    t_leg_start = 0.05 * duration  # Start time for the legs
    t_leg_end = 0.95 * duration  # End time for the legs
    t_cross_start = 0.35 * duration  # Start time for the crossbar
    t_cross_end = 0.65 * duration  # End time for the crossbar
    
    # Generate the legs of 'A'
    leg_indices = np.logical_and(t >= t_leg_start, t <= t_leg_end)
    signal[leg_indices] += np.sin(2 * np.pi * freq_leg1 * t[leg_indices])
    signal[leg_indices] += np.sin(2 * np.pi * freq_leg2 * t[leg_indices])
    
    # Generate the crossbar of 'A'
    crossbar_indices = np.logical_and(t >= t_cross_start, t <= t_cross_end)
    signal[crossbar_indices] += np.sin(2 * np.pi * freq_crossbar * t[crossbar_indices])
    
    # Normalize the signal
    signal = signal / np.max(np.abs(signal))
    
    return signal

def main():
    sample_rate = 44100  # Sample rate in Hz
    duration = 0.25  # Duration in seconds to make the 'A' more distinct
    freq_min, freq_max = 14000, 16500  # Frequency range to encode 'A'
    
    # Generate the improved signal
    signal_A_improved = encode_visual_A(sample_rate, duration, freq_min, freq_max)
    
    # Save the signal to a WAV file
    encoded_file_path_improved = 'encoded_A_improved.wav'
    wavfile.write(encoded_file_path_improved, sample_rate, signal_A_improved.astype(np.float32))
    
    # Plot the spectrogram to visualize the improved 'A'
    plot_spectrogram(signal_A_improved, sample_rate, 'Spectrogram with Encoded A Improved', cmap='inferno')



    # Example file path
    # audio_file_path = 'encoded_audio_clips/AI_Generated_1_modified.wav'
    
    # # Load the audio file
    # sample_rate, audio_data = wavfile.read(audio_file_path)

    # # Specify the frequency range to modify or isolate
    # freq_min, freq_max = 14000, 16500

    # # Modify audio data to blot out a specific frequency range
    # if audio_data.ndim > 1:  # For stereo files, process the first channel as an example
    #     modified_audio_data = modify_frequency(audio_data[:, 0], sample_rate, freq_min, freq_max)
    # else:
    #     modified_audio_data = modify_frequency(audio_data, sample_rate, freq_min, freq_max)

    # # Isolate a frequency range to reveal hidden messages
    # isolated_audio_data = isolate_frequency_range(audio_data, sample_rate, freq_min, freq_max)

    # # Plot spectrogram of the original audio
    # plot_spectrogram(audio_data, sample_rate, 'Original Audio Spectrogram', cmap='viridis')

    # # Plot spectrogram of the modified audio data
    # plot_spectrogram(modified_audio_data, sample_rate, 'Modified Audio Spectrogram', cmap='inferno')

    # # Plot spectrogram with isolated hidden message
    # plot_spectrogram(isolated_audio_data, sample_rate, 'Isolated Hidden Message Spectrogram', cmap='inferno')

if __name__ == "__main__":
    main()