import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap

def hex_to_rgb(value):
	value = value.lstrip('#')
	return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))

ntrees = list(range(50, 250, 50))
maxdepth = list(range(5, 45, 10))

directory = "profiling_results"
listofdicts = list()
for ntree in ntrees:
	for md in maxdepth:
		mydict = dict()
		with open(f"{directory}/memUsage_profiling_test_ntrees{ntree}_maxdepth{md}.txt", "r") as fin:
			
			for line in fin:
				line = line.strip()
				if line:
					mydict["N Estimators"] = ntree
					mydict["Max Depth"] = md
					mydict["RAM Usage"] = line

		listofdicts.append(mydict)

df = pd.DataFrame(listofdicts)
df["RAM Usage"] = df["RAM Usage"].astype(float)
df["RAM Usage"] *= 0.001
df["RAM Usage"] = df["RAM Usage"].round(2)

# Wes Anderson Fantastic Fox 1 palette from GitHub page
bright_painterly = ["#061283", "#FD3C3C", "#FFB74C", "#138D90"]
cheerful_brights = ["#258039", "#F5BE41", "#31A9B8", "#CF3721"]
cpal = sns.color_palette(bright_painterly).as_hex()

sns.barplot(data=df, x="N Estimators", y="RAM Usage", hue="Max Depth", palette=cpal)

sns.despine()

plt.xlabel("Number of RF Trees")
plt.ylabel("RAM Usage (Gb)")

plt.savefig("profiling/ntrees_maxdepth_profiling.pdf", bbox_inches="tight")





