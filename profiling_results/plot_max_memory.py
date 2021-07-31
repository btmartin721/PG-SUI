import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

ntrees = list(range(50, 250, 50))
maxdepth = list(range(5, 45, 10))

print(ntrees)
print(maxdepth)

prof_list = list()
for ntree in ntrees:
	for md in maxdepth:
		with open(f"profiling_results/memUsage_profiling_test_ntrees{ntree}_maxdepth{md}.txt", "r") as fin:

			for line in fin:
				line = line.strip()
				if line:
					



fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1)





