import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0, 0.30340382710933683, 0.3310104872369766, 0.34348129745364187, 0.35091807383656504])
y2 = np.array([0, 0.3813847729873657, np.nan, np.nan, np.nan])
y3 = np.array([0, 0.39953661878108976, np.nan, np.nan, np.nan])
y4 = np.array([0, 0.4145655731010437, np.nan, np.nan, np.nan])
y5 = np.array([0, 0.4117909865605831, np.nan, np.nan, np.nan])
y6 = np.array([0, 0.4195622387599945, 0.45833434972524645, 0.477471009747982, 0.4880324535870552])
y7 = np.array([0, 0.4107974250018597, 0.44753125154256823, 0.46352191767930984, 0.4760648959302902])
y8 = np.array([0, 0.39083645319104193, np.nan, np.nan, np.nan])

l1, = plt.plot(x, y1, color='green', label='# layers = 1 (120MB)')
l2, = plt.plot(x, y2, color='orange', label='# layers = 2 (147MB)')
l3, = plt.plot(x, y3, color='orange', label='# layers = 2, recursive (120MB)')
l4, = plt.plot(x, y4, color='magenta', label='# layers = 3 (174MB)')
l5, = plt.plot(x, y5, color='magenta', label='# layers = 3, recursive (120MB)')
l6, = plt.plot(x, y6, color='blue', label='# layers = 6 (256MB)')
l7, = plt.plot(x, y7, color='red', label='# layers = 6, recursive (120MB)')
l8, = plt.plot(x, y8, color='purple', label='# layers = 12, recursive (120MB)')


plt.xticks(np.arange(min(x), max(x)+1, 1.0))

plt.xlabel("number of oscar en train samples (x1000000)")
plt.ylabel("Masked LM test accuracy (Top-1)")
plt.title("RoBERTa Recursive Transformer")

plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8])

plt.show()
