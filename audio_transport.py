from scipy.signal import stft, istft
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyaudio
from tqdm import tqdm
from time import time

SAMPLERATE = 48000
AUDIO_FILE1 = "440sine48k.wav"
AUDIO_FILE2 = "440saw48k.wav"

# def timing(f):
#     @wraps(f)
#     def wrap(args, **kw):
#         ts = time()
#         result = f(args, **kw)
#         te = time()
#         print 'func:%r args:[%r, %r] took: %2.4f sec' % \
#           (f.name, args, kw, te-ts)
#         return result
#     return wrap

class AudioPlayer:

	def __init__(self, audio_array):
		self.index = 0
		self.arr = audio_array

	def audio_callback(self, in_data, frame_count, time_info, status):
		# TODO: Loop audio_output_array (can we edit audio_output_array asynchronously?)
		out = self.arr[self.index:frame_count + self.index]
		self.index = (self.index + frame_count) % (self.arr.shape[0] // 2)
		return (out, pyaudio.paContinue)

def compute_transport_matrix(X_nT, Y_nT):	
	# Only use magnitude (for now).
	X_nT = np.abs(X_nT)
	Y_nT = np.abs(Y_nT)

	# initialize transport algorithm
	num_bins = X_nT.shape[0]
	T = X_nT.shape[1]

	#PI_nnT = np.zeros((n, n, T)) # policy we are solving for
	PI_T = [[] for _ in range(T)]
	normsx_T = np.zeros(T)
	normsy_T = np.zeros(T)


	# Audio Tranport Algo for each STFT Bin
	for t in range(T):

		i, j = 0, 0

		# get normalized magnitude vectors
		normsx_T[t] = np.linalg.norm(X_nT[:, t])
		normsy_T[t] = np.linalg.norm(Y_nT[:, t])

		X_ = X_nT[:, t] / normsx_T[t]
		Y_ = Y_nT[:, t] / normsy_T[t]

		px, py = X_[i], Y_[j]

		# Audio Transport Algorithm
		while True:
			if px < py:
				PI_T[t].append((i, j, px))
				i += 1

				if i >= num_bins:
					break

				py = py - px
				px = X_[i]
			else:
				PI_T[t].append((i, j, py))
				j += 1

				if j >= num_bins:
					break

				px = px - py
				py = Y_[j]

	return PI_T, normsx_T, normsy_T

# depends on a few of the variables we just calculated
def calculate_interpolation(k, PI_T, wx, normsx_T, normsy_T):
	START_TIME = time()
	T = len(PI_T)
	Z_nT = np.zeros_like(X)
	# TODO: also deal with normalization "scaling is interpolated linearly over the interpolation"
		# this means that for a value k, and timestep t, 
	for t in tqdm(range(T)):
		for i, j, PI_ij in PI_T[t]:
			w = (1-k) * wx[i] + k*wx[j]
			w_index = w * freq_to_idx
			w_index1 = int(w_index)
			# Z_nT[w_index1] += PI_nnT[i][j]
			w_alpha = w_index - w_index1
			Z_nT[w_index1, t] += (1 - w_alpha) * PI_ij
			if w_alpha != 0: # if its 0 we may get an index out of bounds
				Z_nT[w_index1 + 1, t] += w_alpha * PI_ij
	
	# normalize for loudness
	for t in range(T):
		Z_nT[:, t] *= (1 - k) * normsx_T[t] + k * normsy_T[t]

	t, output = istft(Z_nT, fs=SAMPLERATE, window='hann', nperseg=1024, nfft=fft_size)
	END_TIME = time()
	print("COMPUTE INTERPOLATION TOTAL TIME:", END_TIME - START_TIME)
	return output



# load audio
audiox, srx = librosa.load(AUDIO_FILE1, sr=SAMPLERATE)
audioy, sry = librosa.load(AUDIO_FILE2, sr=SAMPLERATE)


# For audio samples of different lengths: pad by wrapping wrap
if len(audiox) > len(audioy):
	audioy = np.pad(audioy, (0, len(audiox) - len(audioy)), mode='wrap')
elif len(audioy) > len(audiox):
	audiox = np.pad(audiox, (0, len(audioy) - len(audiox)), mode='wrap')

audio_length = audiox.shape[0]

fft_size = 2048
num_bins = fft_size // 2 + 1


# get stft
START_TIME = time()

wx, tx, X = stft(audiox, fs=SAMPLERATE, window='hann', nperseg=1024, nfft=fft_size)
wy, ty, Y = stft(audioy, fs=SAMPLERATE, window='hann', nperseg=1024, nfft=fft_size)
# T = tx.shape[0] # Max number of FFT steps
END_TIME = time()
print("STFT TIME:", END_TIME - START_TIME)


# freq * freq_to_idx = idx into w array
freq_to_idx = (num_bins - 1) / (wx[-1])

# sanity check
assert wx.shape[0] == wy.shape[0]
assert [a == b for a, b in zip(wx, wy)] # same frequency bins
assert tx.shape[0] == ty.shape[0]
assert [a == b for a, b in zip(tx, ty)] # same time bins
	

# TODO: Find a profile - find a look at which part is taking so long
# TODO: Try this on dummy spectrums, and just 1 spectrums
START_TIME = time()
PI_T, normsx_T, normsy_T = compute_transport_matrix(X,Y)
END_TIME = time()
print("COMPUTE TRANSPORT MATRIX TIME:", END_TIME - START_TIME)


k = float(input("enter interpolation value (0 to 1, -1 to exit): "))

aoa = calculate_interpolation(k, PI_T, wx, normsx_T, normsy_T)

audio_output_array = np.concatenate([aoa, aoa])


pya = pyaudio.PyAudio()

player = AudioPlayer(audio_output_array)

# init audio stream
stream = pya.open(format=pya.get_format_from_width(width=4), 
				channels=1, 
				rate=SAMPLERATE, 
				output=True, 
				stream_callback=player.audio_callback
			)
frames_per_buffer = stream._frames_per_buffer

stream.start_stream()

# loop unless user inputs -1
while k != -1 and stream.is_active():
	k = float(input("enter interpolation value (0 to 1, -1 to exit): "))
	# TODO: Update audio_output_array and calculate new interpolation
	aoa = calculate_interpolation(k, PI_T, wx, normsx_T, normsy_T)
	player.arr = np.concatenate([aoa, aoa])
	# time.sleep(0.5)

stream.stop_stream()
stream.close()
pya.terminate()