import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0, 0.4007114856934547, 0.4485383033657074, 0.4697808350682259, 0.4805689078116417])
y2 = np.array([0, 0.4195622387599945, 0.45833434972524645, 0.477471009747982, 0.4880324535870552])
y3 = np.array([0, 0.32205438960790633, 0.34977506301760675, 0.3640139799118042, 0.3723136756336689])

l1, = plt.plot(x, y1, color='r', label='# layers = 6, shared layer params [recursive] (249MB)')
l2, = plt.plot(x, y2, color='b', label='# layers = 6 (256MB)')
l3, = plt.plot(x, y3, color='g', label='# layers = 1 (249MB)')
#plt.plot(x, y1, color='r', label='sharedLayerWeights')
#plt.plot(x, y2, color='b', label='wosharedLayerWeights')
plt.xticks(np.arange(min(x), max(x)+1, 1.0))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer (# params normalised)")

plt.legend(handles=[l1, l2, l3])

plt.show()
