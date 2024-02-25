import wave
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os

def plot_audio_spectrum(wav_file_path, time_range=None, frequency_range=None, nfft=1024, cmap='viridis'):
    """
    Plots the spectrogram of an audio file specified by 'wav_file_path'.
    
    This function opens a WAV file, reads its frames, and then converts these frames into a NumPy array to represent the audio signal. 
    It uses the matplotlib library to plot a spectrogram of the audio signal, which is a visual representation of the spectrum of 
    frequencies in the audio signal as they vary with time. The spectrogram is plotted with the time on the x-axis and frequency on the y-axis.
    
    Parameters:
    - wav_file_path: Path to the WAV file to be analyzed.
    - time_range: Optional tuple specifying the time range (start, end) in seconds to plot. If None, plots the entire duration.
    - frequency_range: Optional tuple specifying the frequency range (low, high) in Hz to plot. If None, plots the full frequency range captured.
    - nfft: Number of FFT points; higher values provide finer frequency resolution. Default is 1024.
    - cmap: Colormap for the spectrogram. Default is 'viridis'.
    
    The function displays the spectrogram using the specified colormap and adjusts the plot limits based on the provided time and frequency ranges, if any.
    """
    # Open the WAV file
    with wave.open(wav_file_path, 'rb') as wav_file:
        # Get the audio sample rate and number of frames
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        # Read all frames
        frames = wav_file.readframes(num_frames)

    # Convert frames to a NumPy array of samples
    samples = np.frombuffer(frames, dtype=np.uint8)  # Use uint8 for 8 bits per sample

    # Plot the spectrum of the audio file with adjusted parameters
    plt.figure(figsize=(10, 4))
    plt.specgram(samples, NFFT=nfft, Fs=sample_rate, cmap=cmap)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram of the Audio File')

    # Set x-axis limits based on the specified time range
    if time_range:
        plt.xlim(time_range)

    # Set y-axis limits based on the specified frequency range
    if frequency_range:
        plt.ylim(frequency_range)

    plt.show()

def text_to_binary(message):
    return ''.join(format(ord(char), '08b') for char in message)

def embed_hidden_message(wav_file_path, message):
    # Check if the file exists
    if not os.path.exists(wav_file_path):
        raise ValueError(f"The file {wav_file_path} does not exist.")

    # Convert the message to binary
    binary_message = text_to_binary(message)
    message_length = len(binary_message)
    
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file_path)
    
    # Ensure the data is mono
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel
    
    # Apply FFT
    freq_data = fft(data)
    freq_data_len = len(freq_data)
    
    # Frequency resolution and range
    freq_resolution = sample_rate / freq_data_len
    start_freq = 14500
    end_freq = 16000
    start_index = int(start_freq / freq_resolution)
    end_index = int(end_freq / freq_resolution)
    available_slots = end_index - start_index
    
    # Check if we have enough slots to encode the message
    if message_length > available_slots:
        raise ValueError("Message too long to encode in the given frequency range.")
    
    # Encode the message by altering the amplitude
    for i, bit in enumerate(binary_message):
        index = start_index + i
        if bit == '1':
            # Increase the amplitude to encode '1'
            freq_data[index] *= 10  # Example amplification factor; adjust as needed
        # No else clause; we leave '0's as they are for simplicity
    
    # Apply inverse FFT
    modified_data = ifft(freq_data)
    modified_data_real = np.real(modified_data).astype(np.int16)
    
    # Save the modified audio
    wavfile.write(wav_file_path + "_modified", sample_rate, modified_data_real)
    print(f"Message embedded and saved to 'modified_{wav_file_path}'.")

def blot_out_frequency_range_explicit(wav_file_path, start_freq, end_freq):
    sample_rate, data = wavfile.read(wav_file_path)
    
    # Handle stereo by selecting the first channel or converting to mono
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono by averaging channels
    
    # Apply FFT
    freq_data = fft(data)
    N = len(data)
    
    # Determine frequency indices
    start_index = int(start_freq / sample_rate * N)
    end_index = int(end_freq / sample_rate * N)
    
    # Explicitly zero out the targeted frequency range
    freq_data[start_index:end_index] = 0 + 0j  # Ensure it's set as a complex number
    
    # Apply inverse FFT and take the real part
    modified_data = np.real(ifft(freq_data)).astype(data.dtype)
    
    # Save the modified audio
    wavfile.write(wav_file_path + "_modified.wav", sample_rate, modified_data)

    # Optionally, plot the result to verify
    plot_audio_spectrum(wav_file_path)

def parse_arguments():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description='Plot the spectrogram of a WAV audio file.')
    parser.add_argument('wav_file_path', type=str, help='Path to the WAV file.')
    parser.add_argument('--time_range', type=float, nargs=2, metavar=('START', 'END'), help='Time range in seconds to plot, e.g., --time_range 0 10')
    parser.add_argument('--frequency_range', type=float, nargs=2, metavar=('LOW', 'HIGH'), help='Frequency range in Hz to plot, e.g., --frequency_range 20 20000')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of FFT points. Default is 1024.')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for the spectrogram. Default is "viridis".')

    return parser.parse_args()

def main():
    blot_out_frequency_range_explicit('encoded_audio_clips/AI_Generated_1_modified.wav', 14500, 16000)


    parser = argparse.ArgumentParser(description="Audio file analysis and manipulation tool")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Create the parser for the "plot" command
    parser_plot = subparsers.add_parser('plot', help='Plot the audio spectrum of a file')
    parser_plot.add_argument("wav_file", help="Path to the WAV file to plot")

    # Create the parser for the "embed" command
    parser_embed = subparsers.add_parser('embed', help='Embed a hidden message in an audio file')
    parser_embed.add_argument("wav_file", help="Path to the WAV file in which to embed the message")
    parser_embed.add_argument("--message", help="The message to embed within the audio file", required=True)

    args = parser.parse_args()

    if args.command == 'plot':
        plot_audio_spectrum(args.wav_file)
    elif args.command == 'embed':
        embed_hidden_message(args.wav_file, args.message)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()