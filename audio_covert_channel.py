import wave
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os
from scipy.signal import spectrogram
from scipy.io.wavfile import write
import math

# Sample rate and time vector setup
sample_rate = 44100  # Standard CD-quality sample rate
time = 0

def square(start_time_tones, end_time_tones, freq1, freq2):

    # Reduced number of frequencies and increased spacing
    frequencies = np.linspace(freq1, freq2, 2)  # Use only two frequencies for simplicity

    audio_signal_continuous_tones = np.zeros_like(time)

    # Generate tones with reduced density
    for f in frequencies:
        for t_idx, t in enumerate(time):
            if start_time_tones <= t < end_time_tones:
                audio_signal_continuous_tones[t_idx] += np.sin(2 * np.pi * f * t)

    return audio_signal_continuous_tones

def read_wav_file(filepath):
    # Read the existing WAV file. This is the file to write the message on top of.
    sample_rate, signal = wavfile.read(filepath)
    signal = signal.astype(np.float32)

    # Ensure signal is in the correct format (mono)
    if len(signal.shape) > 1:
        signal = signal[:, 0] # Take the first channel if stereo

    return signal


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

# ===== X5 =====

def makeX5Y1(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 4) * audioLength * .025), ((currentPlace + 5) * audioLength * .025), 19750, 20000)

def makeX5Y2(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 4) * audioLength * .025), ((currentPlace + 5) * audioLength * .025), 19250, 19500)

def makeX5Y3(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 4) * audioLength * .025), ((currentPlace + 5) * audioLength * .025), 18750, 19000)

def makeX5Y4(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 4) * audioLength * .025), ((currentPlace + 5) * audioLength * .025), 18250, 18500)

def makeX5Y5(audio_signal_continuous_tones, audioLength, currentPlace):
    audio_signal_continuous_tones += square(((currentPlace + 4) * audioLength * .025), ((currentPlace + 5) * audioLength * .025), 17750, 18000)


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

# D
def makeD(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)

    currentTime += 5
    return currentTime

# E
def makeE(audio_signal_continuous_tones, audioLength, currentTime):
    # Side
    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Top
    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y1(audio_signal_continuous_tones, audioLength, currentTime)

    # Middle
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)

    # Bottom
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)
    currentTime += 5
    return currentTime

# F
def makeF(audio_signal_continuous_tones, audioLength, currentTime):
    # Side
    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Top
    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y1(audio_signal_continuous_tones, audioLength, currentTime)

    # Middle
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    currentTime += 5
    return currentTime

# G
def makeG(audio_signal_continuous_tones, audioLength, currentTime):
    # Side
    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Top
    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y1(audio_signal_continuous_tones, audioLength, currentTime)
    # Bottom
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX4Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y3(audio_signal_continuous_tones, audioLength, currentTime)
    currentTime += 5
    return currentTime

# H
def makeH(audio_signal_continuous_tones, audioLength, currentTime):
    # Left
    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Right
    makeX4Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)

    # Middle
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    currentTime += 5
    return currentTime

# I
def makeI(audio_signal_continuous_tones, audioLength, currentTime):
    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)
    currentTime += 5
    return currentTime

def makeJ(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)

  

    currentTime += 5
    return currentTime


def makeK(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)



    currentTime += 5
    return currentTime

def makeL(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)

    currentTime += 5
    return currentTime

def makeM(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y2(audio_signal_continuous_tones, audioLength, currentTime)
   
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)

    makeX4Y2(audio_signal_continuous_tones, audioLength, currentTime)
    

    makeX5Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y5(audio_signal_continuous_tones, audioLength, currentTime)


    currentTime += 6
    return currentTime

def makeN(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y2(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    

    makeX4Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)



    currentTime += 5
    return currentTime

def makeO(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)


    currentTime += 5
    return currentTime

def makeP(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)


    currentTime += 5
    return currentTime

def makeQ(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX4Y5(audio_signal_continuous_tones, audioLength, currentTime)

   
    currentTime += 5
    return currentTime

def makeR(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)


    currentTime += 5
    return currentTime

def makeS(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)



    currentTime += 5
    return currentTime

def makeT(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)

    currentTime += 5
    return currentTime

def makeU(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)



    currentTime += 5
    return currentTime

def makeV(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)


    currentTime += 5
    return currentTime

def makeW(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y4(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)

    makeX4Y4(audio_signal_continuous_tones, audioLength, currentTime)

    makeX5Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX5Y5(audio_signal_continuous_tones, audioLength, currentTime)


    currentTime += 6
    return currentTime

def makeX(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)

    currentTime += 5
    return currentTime

def makeY(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y3(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y3(audio_signal_continuous_tones, audioLength, currentTime)

    currentTime += 5
    return currentTime

def makeZ(audio_signal_continuous_tones, audioLength, currentTime):

    makeX1Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y4(audio_signal_continuous_tones, audioLength, currentTime)
    makeX1Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX2Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y3(audio_signal_continuous_tones, audioLength, currentTime)
    makeX2Y5(audio_signal_continuous_tones, audioLength, currentTime)

    makeX3Y1(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y2(audio_signal_continuous_tones, audioLength, currentTime)
    makeX3Y5(audio_signal_continuous_tones, audioLength, currentTime)


    currentTime += 5
    return currentTime

# Space
def makeSpace(audio_signal_continuous_tones, audioLength, currentTime):
    currentTime += 5
    return currentTime


def embed_hidden_message(message, infile, outfile):
    # Deal with varying message length.
    # The 'time' array needs to be long enough to support the entire message. 
    # The higher it is, the longer is takes, so instead of making it very large we should dynamically adjust it to the message len.
    # There are 8 charatcers in a second. So if we have 25 characters, thats ceil(25 / 8) = 4, 4 seconds of space needed.
    message_seconds = math.ceil(len(message) / 8)
    global time
    time = np.linspace(0, message_seconds, message_seconds * sample_rate)
    
    # Creating a starting square. Tiny and out of sight range
    audio_signal_continuous_tones = square(0, 0.001, 20000, 20001)
    currentTime = 0

    print("\nEncoding message: " + message)
    for letter in message.upper():
        print("\rProgress: Encoding \'" + letter + "\' ", end="")
        match letter:
            case 'A':
                currentTime = makeA(audio_signal_continuous_tones, 1, currentTime)
            case 'B':
                currentTime = makeB(audio_signal_continuous_tones, 1, currentTime)
            case 'C':
                currentTime = makeC(audio_signal_continuous_tones, 1, currentTime)
            case 'D':
                currentTime = makeD(audio_signal_continuous_tones, 1, currentTime)
            case 'E':
                currentTime = makeE(audio_signal_continuous_tones, 1, currentTime)
            case 'F':
                currentTime = makeF(audio_signal_continuous_tones, 1, currentTime)
            case 'G':
                currentTime = makeG(audio_signal_continuous_tones, 1, currentTime)
            case 'H':
                currentTime = makeH(audio_signal_continuous_tones, 1, currentTime)
            case 'I':
                currentTime = makeI(audio_signal_continuous_tones, 1, currentTime)
            case 'J':
                currentTime = makeJ(audio_signal_continuous_tones, 1, currentTime)
            case 'K':
                currentTime = makeK(audio_signal_continuous_tones, 1, currentTime)
            case 'L':
                currentTime = makeL(audio_signal_continuous_tones, 1, currentTime)
            case 'M':
                currentTime = makeM(audio_signal_continuous_tones, 1, currentTime)
            case 'N':
                currentTime = makeN(audio_signal_continuous_tones, 1, currentTime)
            case 'O':
                currentTime = makeO(audio_signal_continuous_tones, 1, currentTime)
            case 'P':
                currentTime = makeP(audio_signal_continuous_tones, 1, currentTime)
            case 'Q':
                currentTime = makeQ(audio_signal_continuous_tones, 1, currentTime)
            case 'R':
                currentTime = makeR(audio_signal_continuous_tones, 1, currentTime)
            case 'S':
                currentTime = makeS(audio_signal_continuous_tones, 1, currentTime)
            case 'T':
                currentTime = makeT(audio_signal_continuous_tones, 1, currentTime)
            case 'U':
                currentTime = makeU(audio_signal_continuous_tones, 1, currentTime)
            case 'V':
                currentTime = makeV(audio_signal_continuous_tones, 1, currentTime)
            case 'W':
                currentTime = makeW(audio_signal_continuous_tones, 1, currentTime)
            case 'X':
                currentTime = makeX(audio_signal_continuous_tones, 1, currentTime)
            case 'Y':
                currentTime = makeY(audio_signal_continuous_tones, 1, currentTime)
            case 'Z':
                currentTime = makeZ(audio_signal_continuous_tones, 1, currentTime)
            case ' ':
                currentTime = makeSpace(audio_signal_continuous_tones, 1, currentTime)
            case _:
                raise ValueError("Unsupported letter: " + letter + ", please only use A-Z.")

    ### Mix the square signal with the signal of the input file ###

    # Read infile
    file_to_overlay = infile
    file_to_overlay_signal = read_wav_file(file_to_overlay) # Get signal of file

    signal1 = file_to_overlay_signal
    signal2 = audio_signal_continuous_tones.astype(np.float32) # Convert to 32bit so file sise isn't so big. Otherwise Toxic jumps to 70mb.

    # Extend the shorter signal by padding with zeros
    # This ensures that length of message doesn't affect the length of the song.
    # Ex. 16 character message doesn't make the song limited to 2 seconds. The rest of the song will be unmodified after the message.
    max_length = max(len(signal1), len(signal2))
    extended_signal1 = np.pad(signal1, (0, max_length - len(signal1)), 'constant')
    extended_signal2 = np.pad(signal2, (0, max_length - len(signal2)), 'constant')
    
    # Make the volume of the sqaure signal much higher so you can see it. Makes brighter on spectrogram.
    # probably creates the worst dog whistle ever
    extended_signal2 = extended_signal2 * 30

    # Mix the signals
    mixed_signal = extended_signal1 + extended_signal2

    # Normalize
    max_val = np.max(np.abs(mixed_signal))
    if max_val > 1:
        mixed_signal = mixed_signal / max_val

    # Export to a WAV file
    write(outfile, sample_rate, mixed_signal.astype(np.float32))
    print("\nDone")

def plot_audio_spectrum(wav_file_path, time_range=None, frequency_range=None, nfft=512, cmap='viridis'):
    """
    Plots the spectrogram of an audio file specified by 'wav_file_path'.
    
    This function reads a WAV file, converts its frames into a NumPy array to represent the audio signal, 
    and uses the matplotlib library to plot a spectrogram of the audio signal, which is a visual representation 
    of the spectrum of frequencies in the audio signal as they vary with time. The spectrogram is plotted with 
    the time on the x-axis and frequency on the y-axis.
    
    Parameters:
    - wav_file_path: Path to the WAV file to be analyzed.
    - time_range: Optional tuple specifying the time range (start, end) in seconds to plot. If None, plots the entire duration.
    - frequency_range: Optional tuple specifying the frequency range (low, high) in Hz to plot. If None, plots the full frequency range captured.
    - nfft: Number of FFT points; higher values provide finer frequency resolution. Default is 1024.
    - cmap: Colormap for the spectrogram. Default is 'plasma'.
    """
    # Read the WAV file
    sample_rate, samples = wavfile.read(wav_file_path)

    # Check if stereo and take one channel
    if samples.ndim == 2:
        samples = samples[:, 0]

    # Generate the spectrogram
    f, t, Sxx = spectrogram(samples, fs=sample_rate, nperseg=nfft)

    plt.figure(figsize=(15, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of the Audio File')
    plt.colorbar(label='Intensity [dB]')

    # Set x-axis limits based on the specified time range
    if time_range:
        plt.xlim(0,time_range)

    # Set y-axis limits based on the specified frequency range
    if frequency_range:
        plt.ylim(frequency_range,20000)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Audio file analysis and manipulation tool")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Create the parser for the "plot" command
    parser_plot = subparsers.add_parser('plot', help='Plot the audio spectrum of a file')
    parser_plot.add_argument("wav_file", help="Path to the WAV file to plot")
    parser_plot.add_argument("--xlim", help="The amount of time in seconds to display", required=False)
    parser_plot.add_argument("--ylim", help="The lower bound of the frequency range to include in the plot", required=False)

    # Create the parser for the "embed" command
    parser_embed = subparsers.add_parser('embed', help='Embed a hidden message in an audio file')
    parser_embed.add_argument("wav_file", help="Path to the WAV file in which to embed the message")
    parser_embed.add_argument("outfile", help="Path to the outfile in which to embed the message")
    parser_embed.add_argument("--message", help="The message to embed within the audio file", required=True)

    args = parser.parse_args()

    if args.command == 'plot':
        if (args.xlim != None) & (args.ylim != None):
            plot_audio_spectrum(args.wav_file, float(args.xlim), float(args.ylim))
        elif args.ylim:
            plot_audio_spectrum(args.wav_file, None,float(args.ylim))
        elif args.xlim:
            plot_audio_spectrum(args.wav_file, float(args.xlim))
        else:
            plot_audio_spectrum(args.wav_file)

    elif args.command == 'embed':
        embed_hidden_message(args.message, args.wav_file, args.outfile)
    else:
        parser.print_help()


main()
