import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sample_rate = 16000
NFFT = 512
nfilt = 32
k = np.arange(257)*31.25
import python_speech_features as fs

my_filterbanks = []
M_min = 47.29
M_max = 2840.03
for i in range(26):
    my_filterbanks.append(M_min+i*((M_max-M_min)/27))

filterbanks = fs.get_filterbanks(nfilt=32, lowfreq=1000)
fig, ax = plt.subplots()
for i in range(nfilt):
    ax.plot(np.arange(257)*31.25, filterbanks[i, :])

    ax.set(xlabel='f in Hz', ylabel='Weight')
    ax.grid()

fig.savefig("test.png")
plt.show()
