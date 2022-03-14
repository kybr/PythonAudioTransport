from scipy.signal import istft, stft
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from scipy.interpolate import interp1d
from tqdm import tqdm
from time import time
# from AudioPlayer import AudioPlayer
from SpectralPointAndMass import SpectralPoint, SpectralMass
import pyaudio

import signal,sys,time
terminate = False

def signal_handling(signum,frame):
	global terminate
	terminate = True

signal.signal(signal.SIGINT,signal_handling)

SAMPLERATE = 48000
AUDIO_FILE1 = "440sine48k.wav"
AUDIO_FILE2 = "440saw48k.wav"
FFT_SIZE = 2048
NPERSEG = 2048

LENGTH = 1 # seconds


class AudioPlayer:

	def __init__(self, audio_array):
		self.index = 0
		self.arr = audio_array

	def audio_callback(self, in_data, frame_count, time_info, status):
		# TODO: Loop audio_output_array (can we edit audio_output_array asynchronously?)
		out = self.arr[self.index:frame_count + self.index]
		self.index = (self.index + frame_count) % (self.arr.shape[0] // 2)
		return (out, pyaudio.paContinue)


def main():

	# load audio. we are going to interpolate between audiox and audioy
	audiox, audioy, audio_length = load_audio(AUDIO_FILE1, AUDIO_FILE2, SAMPLERATE)

	# init audio player
	player = AudioPlayer(np.zeros(audio_length))

	pya = pyaudio.PyAudio()

	# init audio stream
	stream = pya.open(format=pya.get_format_from_width(width=4), 
					channels=1, 
					rate=SAMPLERATE, 
					output=True, 
					stream_callback=player.audio_callback)

	frames_per_buffer = stream._frames_per_buffer

	stream.start_stream()

	while stream.is_active():

		if terminate:
			break

		interpolation_factor = uniform(0, 1)
		#interpolation_factor = float(input("enter interpolation value (0 to 1): "))

		# In the c++ implementation, we MAY need to do the whole analysis at every audio callback

		# convert to spectral domain
		spectral_points_x_T = analyze(audiox) # frames x spectral points
		spectral_points_y_T = analyze(audioy) 

		# when I use "T", that refers to number of windows in STFT
		T = len(spectral_points_x_T)
		N = len(spectral_points_x_T[0])

		# initialize phase list and interpolated output list
		# we store the phase of the previous frame for reconstruction purposes
		phases = np.zeros(len(spectral_points_x_T[0]))
		spectral_points_interp_T = []

		# for each STFT time frame, perform optimal transport based interpolation
		test_freqs = [p.freq for p in spectral_points_x_T[30]][:N//10]
		test_freq_reassigned = [p.freq_reassigned for p in spectral_points_x_T[30]][:N//10]
		# plt.plot(test_freqs, test_freqs)
		# plt.plot(test_freqs, test_freq_reassigned)
		# plt.plot([p.freq_reassigned for p in spectral_points_x_T[1]], [np.abs(p.value) for p in spectral_points_x_T[1]])
		# plt.show()
		for t in (range(T)):
		#for t in tqdm(range(T)):
			phases, output_points = interpolate(
				spectral_points_x_T[t],
				spectral_points_y_T[t],
				phases,
				FFT_SIZE,
				interpolation_factor
			)
			spectral_points_interp_T.append(
				output_points
			)
		# synthesize audio from interpolated spectrum, replace audio in player
		output = synthesize(spectral_points_interp_T)

		# normalize Output
		if (np.max(np.abs(output)) != 0):
			output = output / (2*np.max(np.abs(output)))

		# sf.write("output.wav", output, SAMPLERATE)

		player.arr = np.concatenate([output, output])

	stream.stop_stream()
	stream.close()
	pya.terminate()


def load_audio(file1, file2, sr):
	# load audio
	audiox, srx = librosa.load(file1, sr=sr)
	audioy, sry = librosa.load(file2, sr=sr)

	# For audio samples of different lengths: pad by wrapping wrap
	if len(audiox) > len(audioy):
		audioy = np.pad(audioy, (0, len(audiox) - len(audioy)), mode='wrap')
	elif len(audioy) > len(audiox):
		audiox = np.pad(audiox, (0, len(audioy) - len(audiox)), mode='wrap')

	audio_length = audiox.shape[0]

	return audiox[:LENGTH * sr], audioy[:LENGTH * sr], LENGTH * sr

def analyze(audio):
	# Audio to Spectral Points

	# We compute 3 different STFTs with 3 different windows
	windows = [hann(NPERSEG), time_weighted_hann(NPERSEG), derivative_hann(NPERSEG)]
	# plt.plot(windows[0])
	# plt.plot(windows[1])
	# plt.show()
	# freqs, times, X = stft(audio, fs=SAMPLERATE, window=windows[0], nperseg=NPERSEG, nfft=FFT_SIZE)
	X = librosa.stft(audio, n_fft=FFT_SIZE, hop_length=NPERSEG//2, win_length= NPERSEG, window=windows[0], center=False)
	# _, _, X_time_weighted_hann = stft(audio, fs=SAMPLERATE, window=windows[1], nperseg=NPERSEG, nfft=FFT_SIZE)
	X_time_weighted_hann = librosa.stft(audio, n_fft=FFT_SIZE, hop_length=NPERSEG//2, win_length= NPERSEG, window=windows[1], center=False)
	# _, _, X_derivative_hann = stft(audio, fs=SAMPLERATE, window=windows[2], nperseg=NPERSEG, nfft=FFT_SIZE)
	X_derivative_hann = librosa.stft(audio, n_fft=FFT_SIZE, hop_length=NPERSEG//2, win_length= NPERSEG, window=windows[2], center=False)
	# T, N = len(times), len(freqs) # T = num_frames, N + num_bins
	N, T = X.shape
	freqs = np.arange(N) * SAMPLERATE / (2 * N)

	output_points = []

	# For each time frame
	for t in (range(T)):
	#for t in tqdm(range(T)):
		output_points.append([])

		# get the time of the center of this frame
		center_time = (1 + 2*t) * NPERSEG / (4 * SAMPLERATE)
		# For each frequency bin
		for n in range(N):
			# p = SpectralPoint(X[n][t], center_time, freqs[n])
			# This is a weird frequency value, but I think its in rad/sec instead of oscillations/sec
			p = SpectralPoint(X[n][t], 2 * np.pi * n * SAMPLERATE / FFT_SIZE)

			# Compute how the frequency and time reassignment
			# going off of https://arxiv.org/pdf/0903.3080.pdf on page 20 eq. (64) and (65)
			# slightly different from from https://github.com/sportdeath/audio_transport/blob/24171de435bc5c07ad57433907581399b81fd6b3/src/spectral.cpp#L162
			# pretty sure the github implementation is a bit wrong - idk though
			conj_over_norm = np.conj(X[n][t]) / (np.abs(X[n][t])**2)
			# dphase_domega = np.real(X_time_weighted_hann[n][t] * conj_over_norm)
			dphase_dt = -np.imag(X_derivative_hann[n][t] * conj_over_norm)
			# print(X_time_weighted_hann[n][t] * conj_over_norm)
			# Compute the reassigned time and frequency
			# p.time_reassigned = p.time + dphase_dt
			p.freq_reassigned = p.freq + dphase_dt

			# add p to output list
			output_points[t].append(p)

	return output_points


def interpolate(points_x_N, points_y_N, phases, window_size, interp_factor):
	# Interpolate between Spectral Points using Optimal Transport
	# Return Spectral Points

	assert len(points_x_N) == len(points_y_N) == len(phases)

	N = len(points_x_N) # number of bins

	# turn spectral points (single frequency bins) into spectral masses (groups of bins)
	masses_x = group_spectrum(points_x_N)
	masses_y = group_spectrum(points_y_N)

	# get transport matrix
	PI = transport_matrix(masses_x, masses_y)

	# init output array
	interpolated = []
	for i in range(N):
		interpolated.append(SpectralPoint())
		interpolated[i].freq = points_x_N[i].freq

	# initialize new amplitudes and phases
	new_amplitudes = np.zeros_like(phases)
	new_phases = np.zeros_like(phases)

	# perform interpolation
	for i, j, mass in PI:
		mass_x = masses_x[i]
		mass_y = masses_y[j]

		# calculate new bin and frequency
		# TODO: Is it better to interpolate the bin rather than rounding
		interp_bin = round(
			(1 - interp_factor) * mass_x.center_bin +
			interp_factor * mass_y.center_bin
		)

		# adjust interpolation factor for rounding
		interp_rounded = interp_factor
		if mass_x.center_bin != mass_y.center_bin:
			interp_rounded = (interp_bin - mass_x.center_bin) / (mass_y.center_bin - mass_x.center_bin)

		# get the interpolated frequency
		interp_freq = (1 - interp_rounded) * points_x_N[mass_x.center_bin].freq_reassigned + \
						interp_rounded * points_y_N[mass_y.center_bin].freq_reassigned

		center_phase = phases[interp_bin] + (interp_freq * window_size / 2) / 2 - np.pi * interp_bin

		new_phase = center_phase + (interp_freq * window_size / 2) / 2 + np.pi * interp_bin

		# # UNCOMMENT FOR HORIZONTAL INCOHERENCE
		# center_phase = np.angle(points_x_N[interp_bin].value)

		# Place masses filling up new phases and interpolated
		place_mass(
			mass_x, 
			interp_bin, 
			(1 - interp_factor) * mass / mass_x.mass,
			interp_freq,
			center_phase,
			points_x_N,
			interpolated,
			new_phase,
			new_phases,
			new_amplitudes
		)

		place_mass(
			mass_y, 
			interp_bin, 
			interp_factor * mass / mass_y.mass,
			interp_freq,
			center_phase,
			points_y_N,
			interpolated,
			new_phase,
			new_phases,
			new_amplitudes
		)

	# fill the phases with new phases
	for i in range(N):
		phases[i] = new_phases[i]

	return phases, interpolated

		
def place_mass(mass, center_bin, scale, interp_freq, center_phase, inp, out, next_phase, phases, amplitudes):
	phase_shift = center_phase - np.angle(inp[mass.center_bin].value)

	for i in range(mass.left_bin, mass.right_bin):
		# compute loc in new array
		new_i = i + center_bin - mass.center_bin
		if (new_i >= 0 and new_i < len(out)):
			phase = phase_shift + np.angle(inp[i].value)
			mag = scale * abs(inp[i].value)
			# add complex value of correct mag and phase to output array
			out[new_i].value += mag * np.exp(1j*phase)

			if (mag > amplitudes[new_i]):
				amplitudes[new_i] = mag
				phases[new_i] = next_phase
				out[new_i].freq_reassigned = interp_freq



def group_spectrum(spectral_points_N):
	# spectral points to spectral mass
	N = len(spectral_points_N)

	# get total mass to normalize with
	total_mass = sum([abs(p.value) for p in spectral_points_N])

	# initialize first mass
	masses = [SpectralMass()]

	sign = spectral_points_N[0].freq_reassigned > spectral_points_N[0].freq

	for i, p in enumerate(spectral_points_N):
		current_sign = p.freq_reassigned > p.freq


		# # To hear vertical incoherence uncomment this:
		# sign = False
		# current_sign = not sign

		if current_sign != sign:
			if sign:
				# We are falling - going from freq_reassigned greater to freq_reassigned less
				# falling is center bin - red dot in paper diagram
				# https://arxiv.org/pdf/1906.06763.pdf top right of page 3

				# this is an edge case - we will go to the mass we are closest to
				# because sign = True and current_sign = False, both of these will be positive
				left_dist = spectral_points_N[i - 1].freq_reassigned - spectral_points_N[i - 1].freq
				right_dist = p.freq - p.freq_reassigned

				# Go to the closer side
				if (left_dist < right_dist):
					masses[-1].center_bin = i - 1
				else:
					masses[-1].center_bin = i
			else:
				# else we are rising - rising is blue line in paper diagram
				# we are also notably at the end of a bin

				# here we will compute the mass
				# add up all the masses from the bin so far
				for j in range(masses[-1].left_bin, i):
					masses[-1].mass += abs(spectral_points_N[j].value)

				if masses[-1].mass > 0:
					# normalize
					masses[-1].mass /= total_mass

					# set the right bin end of the mass
					masses[-1].right_bin = i

					# construct the next mass
					new_mass = SpectralMass()
					new_mass.left_bin = i
					new_mass.center_bin = i
					masses.append(new_mass)

		# send current_sign to previous sign variable
		sign = current_sign

	# finish final mass
	masses[-1].right_bin = N
	for j in range(masses[-1].left_bin, N):
		masses[-1].mass += abs(spectral_points_N[j].value)

	# normalize final mass
	if masses[-1].mass > 0:
		masses[-1].mass /= total_mass

	return masses

def transport_matrix(X, Y):
	# X and Y are lists of spectral masses for the spectrums we are interpolating betweens
	# return sparse array PI (list) of (i, j, mass) pairs of transport instructions
	PI = []

	i, j = 0, 0

	# px (greek p [row]) is mass left in bin i
	px, py = X[i].mass, Y[j].mass

	# Audio Transport Algorithm
	# Runtime O(N)
	while True:
		if px < py:
			PI.append((i, j, px))

			py = py - px # subtract off the mass we use
			
			i += 1
			if i >= len(X):
				break
			
			px = X[i].mass
		else: # symmetric
			PI.append((i, j, py))

			px = px - py # subtract off the mass we use

			j += 1
			if j >= len(Y):
				break
			py = Y[j].mass

	return PI



def synthesize(spectral_points_TN):
	# Spectral Points to Audio
	# assume spectral points are a 2D array of T by N
	Z_NT = np.array([[p.value for p in t] for t in spectral_points_TN]).T

	# perform istft
	t, output = istft(Z_NT, fs=SAMPLERATE, window="hann", nperseg=NPERSEG, nfft=FFT_SIZE)
	# output = librosa.istft(Z_NT, n_fft=FFT_SIZE, hop_length=NPERSEG//2, win_length= NPERSEG, window=hann(NPERSEG), center=False)
	return output.astype(np.float32)

def hann(length):
	# return a hann window
	# t = np.arange(-(length - 1) // 2, 1 + (length - 1) // 2)
	# return 0.5 + 0.5 * np.cos(2 * np.pi * t/(length - 1))
	return np.sin(np.pi * np.arange(length) / length)**2

def time_weighted_hann(length):
	# return a time weighted hann window of length
	# t = np.arange(-(length - 1) // 2, 1 + (length - 1) // 2)
	# return t * hann(length) / SAMPLERATE
	return np.arange(length) * hann(length) / SAMPLERATE

def derivative_hann(length):
	# return a time derivative of hann window of length
	# length = sr * window_length_in_seconds
	# we are deriving w.r.t t = np.arange(length) / sr 
	# derivative uses law: 2sin(x)cos(x) = sin(2x)

	return np.pi * SAMPLERATE * np.sin(2 * np.pi * np.arange(length) / length) / length

import cProfile
cProfile.run('main()')

if __name__ == "__main__":
	main()
