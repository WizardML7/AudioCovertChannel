import wave
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os
from scipy.signal import spectrogram
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Sample rate and time vector setup
sample_rate = 44100  # Standard CD-quality sample rate
time = np.linspace(0, 1, sample_rate)  # 1 second of audio


def square(start_time_tones, end_time_tones, freq1, freq2):

    # Reduced number of frequencies and increased spacing
    frequencies = np.linspace(freq1, freq2, 2)  # Use only two frequencies for simplicity

    audio_signal_continuous_tones = np.zeros_like(time)

    # Generate tones with reduced density
    for f in frequencies:
        for t_idx, t in enumerate(time):
            if start_time_tones <= t < end_time_tones:
                audio_signal_continuous_tones[t_idx] += np.sin(2 * np.pi * f * t)

    # Normalize the audio signal
    audio_signal_continuous_tones /= np.max(np.abs(audio_signal_continuous_tones))
    return audio_signal_continuous_tones

# ===== X1 =====


def makeX1Y1(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square((currentPlace * audioLength * .025), ((currentPlace + 1) * audioLength * .025), 19750, 20000)

def makeX1Y2(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square((currentPlace * audioLength * .025), ((currentPlace + 1) * audioLength * .025), 19250, 19500)

def makeX1Y3(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square((currentPlace * audioLength * .025), ((currentPlace + 1) * audioLength * .025), 18750, 19000)

def makeX1Y4(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square((currentPlace * audioLength * .025), ((currentPlace + 1) * audioLength * .025), 18250, 18500)

def makeX1Y5(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square((currentPlace * audioLength * .025), ((currentPlace + 1) * audioLength * .025), 17750, 18000)


# ===== X2 =====

def makeX2Y1(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 1) * audioLength * .025), ((currentPlace + 2) * audioLength * .025), 19750, 20000)

def makeX2Y2(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 1) * audioLength * .025), ((currentPlace + 2) * audioLength * .025), 19250, 19500)

def makeX2Y3(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 1) * audioLength * .025), ((currentPlace + 2) * audioLength * .025), 18750, 19000)

def makeX2Y4(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 1) * audioLength * .025), ((currentPlace + 2) * audioLength * .025), 18250, 18500)

def makeX2Y5(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 1) * audioLength * .025), ((currentPlace + 2) * audioLength * .025), 17750, 18000)


# ===== X3 =====

def makeX3Y1(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 2) * audioLength * .025), ((currentPlace + 3) * audioLength * .025), 19750, 20000)

def makeX3Y2(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 2) * audioLength * .025), ((currentPlace + 3) * audioLength * .025), 19250, 19500)

def makeX3Y3(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 2) * audioLength * .025), ((currentPlace + 3) * audioLength * .025), 18750, 19000)

def makeX3Y4(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 2) * audioLength * .025), ((currentPlace + 3) * audioLength * .025), 18250, 18500)

def makeX3Y5(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 2) * audioLength * .025), ((currentPlace + 3) * audioLength * .025), 17750, 18000)



# ===== X4 =====

def makeX4Y1(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 3) * audioLength * .025), ((currentPlace + 4) * audioLength * .025), 19750, 20000)

def makeX4Y2(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 3) * audioLength * .025), ((currentPlace + 4) * audioLength * .025), 19250, 19500)

def makeX4Y3(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 3) * audioLength * .025), ((currentPlace + 4) * audioLength * .025), 18750, 19000)

def makeX4Y4(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 3) * audioLength * .025), ((currentPlace + 4) * audioLength * .025), 18250, 18500)

def makeX4Y5(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 3) * audioLength * .025), ((currentPlace + 4) * audioLength * .025), 17750, 18000)




# A
def makeA(audio_signal_continuous_tones, audioLength, currentTime):
    # Left vertical line
    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Right vertical line
    makeX4Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Middle horizontal Line
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)

    # Top horizontal Line
    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)

    # Increment for next letter
    currentTime += 5
    return currentTime


# B
def makeB(audio_signal_continuous_tones, audioLength, currentTime):
    # Top horizontal Line
    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)

    # Bottom horizontal Line
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Middle horizontal Line
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y3(audio_signal_continuous_tones, audioLength, currentTime)

    # Left vertical line
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)

    # Right side connectors
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y4(audio_signal_continuous_tones, audioLength, currentTime)

    # Increment for next letter
    currentTime += 5
    return currentTime


# C
def makeC(audio_signal_continuous_tones, audioLength, currentTime):
    # Top horizontal Line
    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y1(audio_signal_continuous_tones, audioLength, currentTime)

    # Bottom horizontal Line
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Left vertical line
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)

    # Increment for next letter
    currentTime += 5
    return currentTime



def lower_volume(input_file, output_file, volume_scale=5):
    """
    Lowers the volume of a WAV file.
    
    Parameters:
    - input_file: Path to the input WAV file.
    - output_file: Path to save the output WAV file with lower volume.
    - volume_scale: Factor to scale the volume by. Default is 0.5 (reduce volume by half).
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(input_file)

    # Apply volume scaling
    if len(data.shape) == 1:
        # Mono audio
        data = np.int16(data * volume_scale)
    else:
        # Stereo audio (or more channels)
        data = np.int16(data * volume_scale)

    # Write the modified data back to a new WAV file
    wavfile.write(output_file, sample_rate, data)


# Creating a starting square. Tiny and out of sight range
audio_signal_continuous_tones = square(0, 0.001, 20000, 20001)

# TODO: Make audio length not hardcoded (it should work with long audio files once this is done)
currentTime = 0
currentTime = makeA(audio_signal_continuous_tones, 1, currentTime)
currentTime = makeB(audio_signal_continuous_tones, 1, currentTime)
currentTime = makeC(audio_signal_continuous_tones, 1, currentTime)



# Export to a WAV file
wav_file_continuous_tones = "less_dense_continuous_tones.wav"
write(wav_file_continuous_tones, sample_rate, audio_signal_continuous_tones.astype(np.float32))

lower_volume("less_dense_continuous_tones.wav", "less_dense_continuous_tones.wav")

# Generate the spectrogram for the less dense tones
frequencies_continuous, times_continuous, spectrogram_matrix_continuous = spectrogram(audio_signal_continuous_tones, sample_rate)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times_continuous, frequencies_continuous, 10 * np.log10(spectrogram_matrix_continuous), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram with Less Dense Continuous Tones')
plt.colorbar(label='Intensity [dB]')
plt.ylim(0, 20000)
plt.show()
