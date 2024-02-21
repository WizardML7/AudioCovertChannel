import wave
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile
from scipy.fft import fft, ifft

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

def embed_hidden_message(wav_file_path, message):
    # Placeholder for the function to embed a message in the specified frequency range
    print(f"Embedding message in {wav_file_path}... (This functionality is not yet implemented.)")


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