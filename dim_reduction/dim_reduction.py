# Standard library imports
from pathlib import Path
import sys

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

# Custom imports
from utils.misc import timer
from utils.misc import progressbar

class DimReduction:

	def __init__(self, gt, pops, prefix, reps=1, colors=None, palette="Set1"):

		self.gt = gt
		self.pops = pops
		self.prefix = prefix
		self.reps = reps
		self.colors = colors
		self.palette = palette

	def _validate_args(self, _dimreduction, _gt, _pops, _prefix, _reps):

		if _dimreduction is None:
			if _gt is None:
				raise TypeError("The 'gt' keyword argument must be defined if dimreduction=None")
			if _pops is None:
				raise TypeError("The 'pops' keyword argument must be defined if dimreduction=None")
			if _prefix is None:
				raise TypeError("The 'prefix' argument must be defined if dimreduction=None")

			if _reps is None:
				self.reps = 1
			else:
				self.reps = _reps

			gt_df = self._validate_type(_gt)
			self.set_gt(gt_df)

		else: # _dimreduction is not None
			if _gt is not None:
				raise TypeError("The 'dimreduction' and 'gt' arguments cannot both be defined")
			if _pops is not None:
				raise TypeError("The 'dimreduction' and 'pops' arguments cannot both be defined")
			if _prefix is not None:
				raise TypeError("The 'dimreduction and 'prefix' arguments cannot both be defined")

			if _reps is None:
				self.set_reps(_dimreduction.reps)
			else:
				self.reps = _reps
			
			gt_df = self._validate_type(_dimreduction.gt)
			self.set_gt(gt_df)
			self.set_pops(_dimreduction.pops)
			self.set_prefix(_dimreduction.prefix)
			self.set_colors(_dimreduction.colors)
			self.set_palette(_dimreduction.palette)

	def _validate_type(self, X):
		if isinstance(X, np.ndarray):
			df = pd.DataFrame(X)

		elif isinstance(X, list):
			df = pd.DataFrame.from_records(X)
		
		elif isinstance(X, pd.DataFrame):
			df = X.copy()

		else:
			raise TypeError("\nThe genotype data must be a numpy.ndarray, a pandas.DataFrame, or a 2-dimensional list of shape (n_samples, n_sites)! Any of these can be retrieved from the GenotypeData object")

		return df

	def set_gt(self, _gt):
		self.gt = _gt

	def set_pops(self, _pops):
		self.pops = _pops

	def set_prefix(self, _prefix):
		self.prefix = _prefix

	def set_colors(self, _colors):
		self.colors = _colors

	def set_palette(self, _palette):
		self.palette = _palette

	def set_reps(self, _reps):
		self.reps = _reps

	def plot(self, plot_3d=False, show_plot=False, axis1=1, axis2=2, axis3=3, figwidth=6, figheight=6, alpha=1.0, legend=True, legend_inside=False, legend_loc="upper left", marker="o", markersize=6, markeredgecolor="k", markeredgewidth=0.5, labelspacing=0.5, columnspacing=2.0, title=None, title_fontsize=None, markerfirst=True, markerscale=1.0, ncol=1, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, legend_edgecolor="black", facecolor="white", framealpha=0.8, shadow=False):
		"""[Plot PCA results as a scatterplot and save it as a PDF file]

		Args:
			axis1 (int, optional): [First axis to plot]. Defaults to 1.

			axis2 (int, optional): [Second axis to plot]. Defaults to 2.

			axis3 (int, optional): [third axis to plot]. Defaults to 3.


		Raises:
			ValueError: [Must be a supported dimensionality reduction method]
			TypeError: [If pca=True, pca_model argument must also be set]
		"""
		# Path().mkdir() creates all directories in path if any don't exist
		if hasattr(self, "clust_method"):
			plot_dir = "{}_output/{}/{}/plots".format(
				self.prefix, self.method, self.clust_method
			)

		else:
			plot_dir = "{}_output/{}/plots".format(
				self.prefix, 
				self.method
			)

		Path(plot_dir).mkdir(parents=True, exist_ok=True)

		if not hasattr(self, "clust_method"):
			uniq_pops = list(set(self.pops))
			colors = self._get_pop_colors(uniq_pops, self.palette, self.colors)
			pop_df = pd.DataFrame(self.pops, columns=["population"])


		for rep in progressbar(range(self.reps), "{} scatterplot: ".format(self.method)):

			if hasattr(self, "clust_method"):
				uniq_pops = list(set(self.labels[rep]))
				uniq_pops = [x+1 if x >= 0 else x+0 for x in uniq_pops]

				colors = self._get_pop_colors(
					uniq_pops, self.palette, None
				)

				labs = self.labels[rep]
				labs = [x+1 if x >= 0 else x+0 for x in labs]

				pop_df = pd.DataFrame(labs, columns=["population"])

			if self.method == "PCA":
				if self.pca_model is None:
					raise TypeError("pca_model argument must be provided if pca=True")

			fig = plt.figure(figsize=(figwidth, figheight))

			if plot_3d:
				ax = fig.add_subplot(111, projection="3d")

			else:
				ax = fig.add_subplot(1,1,1)

			if self.method == "PCA":
				self._plot_coords(self.coords[rep], axis1, axis2, axis3, plot_3d, ax, pop_df, uniq_pops, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, self.method, model=self.pca_model[rep])

			else:
				self._plot_coords(self.coords[rep], axis1, axis2, axis3, plot_3d, ax, pop_df, uniq_pops, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, self.method)

			if legend:
				if legend_inside:
					if bbox_to_anchor[0] > 1 or \
						bbox_to_anchor > 1:
						print("Warning: bbox_to_anchor was set grater than 1.0 (outside plot margins) but legend_inside was set to True. Setting bbox_to_anchor to (1.0, 1.0)")

				else:
					if bbox_to_anchor[0] < 1 and \
						bbox_to_anchor[1] < 1:
						print("Warning: bbox_to_anchor was set less than 1.0 (inside the plot margins) but legend_inside was set to False. Setting bbox_to_anchor to (1.05, 1.0)")

				ax.legend(loc=legend_loc, labelspacing=labelspacing, columnspacing=columnspacing, title=title, title_fontsize=title_fontsize, markerfirst=markerfirst, markerscale=markerscale, ncol=ncol, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad, edgecolor=legend_edgecolor, facecolor=facecolor, framealpha=framealpha, shadow=shadow)

			plot_fn = "{}/scatterplot_{}.pdf".format(plot_dir, rep+1)

			fig.savefig(plot_fn, bbox_inches="tight")

			if show_plot:
				plt.show()
			
			fig.clf()
			plt.close(fig)

		print("\nSaved {} scatterplot(s) to {}".format(self.method, plot_dir))

	def _plot_coords(self, coords, axis1, axis2, axis3, plot_3d, ax, populations, unique_populations, pop_colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, method, model=None):
		"""[Map colors to populations and make the scatterplot]

		Args:
			coords ([numpy.array]): [pca_coords object returned from scikit-allel PCA or cmds_model or isomds_model objects stored in do_mds()]

			axis1 ([int]): [First axis to plot. Starts at 1]

			axis2 ([int]): [Second axis to plot. Starts at 1]

			axis3 ([int]): [Third axis to plot. Starts at 1]

			plot_3d ([bool]): [True if making 3D plot. False if 2D plot]

			ax ([matplotlib object]): [ax object from matplotlib]

			populations ([pandas.DataFrame]): [pandas.DataFrame with population IDs]

			unique_populations ([list]): [Unique populations in the populations argument]

			pop_colors ([dict]): [Dictionary with unique population IDs as keys and hex-code colors as values]

			alpha ([float]): [Set transparency of points; lower = more transparent. Should be between 0 and 1]

			marker ([str]): [Set the marker shape. See matplotlib.markers documentation].

			markersize ([int]): [Set size of the markers]

			markeredgecolor ([str]): [Set the color of the marker edge]. See matplotlib.pyplot.plot documentation.

			markeredgewidth ([float]): [Set the width of the marker edge]

			method ([str]): [Dimensionality reduction method to use]

			model (scikit-allel object, optional): [Second object returned from scikit-allel PCA. Set model=None if doing cmds or isomds]

		Raises:
			ValueError: [Make sure model argument is set if pca is True]
			ValueError: [Make sure one of pca, cmds, or isomds is True]
		"""
		sns.despine(ax=ax, offset=5)
		axis1_idx = axis1 - 1
		axis2_idx = axis2 - 1
		x = coords[:, axis1_idx]
		y = coords[:, axis2_idx]

		if plot_3d:
			if int(coords.shape[1]) < 3:
				raise ValueError("plot_3d was specified"
								"but there are fewer than 3 coordinate axes!"
				)
			axis3_idx = axis3 - 1
			z = coords[:, axis3_idx]

		for i, pop in enumerate(unique_populations, start=1):

			flt = (populations.population == pop)

			if pop == -1:
				pop_colors[pop] = "k"
				alpha = 1.0
				lab = "Noisy Samples"
				if markersize > 2:
					ms = markersize - 2

				elif markersize > 1:
					ms = markersize - 1

			else:
				lab = pop
				ms = markersize

			if plot_3d:
				ax.plot3D(
							x[flt], 
							y[flt], 
							z[flt],
							marker=marker, 
							linestyle=' ', 
							color=pop_colors[pop], 
							label=lab, 
							markersize=ms, 
							mec=markeredgecolor, 
							mew=markeredgewidth, 
							alpha=alpha
				)

			else:
				ax.plot(
						x[flt], 
						y[flt], 
						marker=marker, 
						linestyle=' ', 
						color=pop_colors[pop], 
						label=pop, 
						markersize=markersize, 
						mec=markeredgecolor, 
						mew=markeredgewidth, 
						alpha=alpha
				)

		if method == "PCA":
			ax.set_xlabel('PC%s (%.1f%%)' % (axis1, model.explained_variance_ratio_[axis1_idx]*100))

			ax.set_ylabel('PC%s (%.1f%%)' % (axis2, model.explained_variance_ratio_[axis2_idx]*100))

			if plot_3d:
				ax.set_zlabel('PC%s (%.1f%%)' % (axis3, model.explained_variance_ratio_[axis3_idx]*100))

		else:
			ax.set_xlabel("{} Axis {}".format(method, axis1))
			ax.set_ylabel("{} Axis {}".format(method, axis2))

			if plot_3d:
				ax.set_zlabel("{} Axis {}".format(method, axis3))

	def _get_pop_colors(self, uniq_pops, palette, colors):
		"""[Get population color codes if colors=None]

		Args:
			uniq_pops ([list]): [Unique population IDs]

			palette ([str]): [Color palette to use. See matplotlib.colors documentation]

			colors ([dict]): [Dictionary with unique population IDs as keys and hex-code colors as values. If colors=None, the color palette is automatically set]

		Raises:
			ValueError: [colors must be equal to the number of unique populations]

		Returns:
			[dict]: [Dictionary with unique population IDs as the keys and hex-code colors as the values]
		"""
		if hasattr(self, "clust_method"):
			colors = None

		if colors is None: # Make a list of hex-coded colors
			colors = dict()
			cmap = plt.get_cmap(palette, len(uniq_pops))

			for i in range(cmap.N):
				rgba = cmap(i)
				colors[uniq_pops[i]] = mcolors.rgb2hex(rgba)

		else:
			if len(colors.keys()) != len(uniq_pops):
				raise ValueError("\nThe colors argument's list length must equal the number of unique populations!")

		return colors

	def save_labels(self, rep, k=None):

		if k is None:
			labs = self.labels[rep]
		else:
			labs = self.labels[rep][k]

		label_dir = \
			"{}_output/{}/{}/labels".format(
				self.prefix, self.method, self.clust_method
		)

		# Makes all directories in path if any don't exist
		Path(label_dir).mkdir(parents=True, exist_ok=True)

		self._write_labels(rep, label_dir, k=k)
		self.pred_labels.append(labs)

	def _write_labels(self, _rep, _label_dir, k=None):

		label_fn = "{}/labels_{}.csv".format(_label_dir, _rep+1)

		if k is None:
			labs = self.labels[_rep]
		else:
			labs = self.labels[_rep][_k]

		try:
			with open(label_fn, "w") as fout:
				for i, lab in enumerate(labs):
					if self.sampleids is None:
						fout.write("{},{}\n".format(i+1, lab))
					else:
						fout.write("{},{}\n".format(self.sampleids[i], lab))
		except OSError:
			print("Could not open or write to file: {}".format(label_fn))

	def _plot_msw_line(self, _silhouettes, _bestk, show_plot, _plot_dir, _rep):

		_fig = plt.figure(figsize=(6, 6))
		_ax = plt.subplot(1, 1, 1)

		# Remove top and right axis lines
		sns.despine(ax=_ax)

		_ax.plot(list(_silhouettes.keys()), list(_silhouettes.values()), linewidth=2.5, color="b", linestyle="-")

		_ax.plot(list(_silhouettes.keys()), list(_silhouettes.values()), marker="o", markersize=12, markerfacecolor="b", label="Mean Silhouette Width")

		_ax.axvline(_bestk, color="r", linestyle="--", linewidth=2.5, label="Optimal K")

		_ax.set_ybound(lower=0, upper=1)
		_ax.set_xlabel("K", fontsize="large")
		_ax.set_ylabel("Mean Silhouette Width", fontsize="large")
		_ax.tick_params(axis="both", labelsize="large")

		_ax.legend()

		# Makes all directories in path if any don't exist
		plot_fn_msw = "{}/msw_lineplot_{}.pdf".format(_plot_dir, _rep+1)

		_fig.savefig(plot_fn_msw, bbox_inches="tight")

		if show_plot:
			plt.show()

		_fig.clf()
		plt.close(_fig)

	def _plot_msw(self, _silhouette_avg, _k, _pp, _rep, **kwargs):

		self._plot_sil_blobs(_silhouette_avg, _k, _rep, kwargs)
		self._plot_clusters(_k, _rep, kwargs)

		plt.suptitle(("PAM Clustering "
						"with MSW Optimal K = {}".format(_k)),
						fontsize=kwargs["plot_title_fontsize"], y=kwargs["sup_title_y"])

		_pp.savefig(bbox_inches="tight")

		if kwargs["show_clusters_plot"]:
			plt.show()

		plt.clf()

	def _plot_sil_blobs(self, _silhouette_avg, _k, _rep, kwargs):

		ax1 = plt.subplot(1, 2, 1)

		# Remove the top and right spines from plots
		sns.despine(ax=ax1, offset=5)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(self.coords[_rep], self.labels[_rep][_k])

		# The 1st subplot is the silhouette plot
		# Silhouettes can range between -1 and 1
		ax1.set_xlim([kwargs["xmin"], 1])
		ax1.set_xbound(lower=kwargs["xmin"], upper=1)

		# The (k+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax1.set_ylim([0, len(self.coords[_rep]) + (_k + 1) * 10])

		y_lower = 10
		yticks = list()
		for i in range(_k):

			label_i = i+1

			# Aggregate the silhouette scores for samples belonging to
			# cluster i, and sort them
			ith_cluster_silhouette_values = \
			sample_silhouette_values[self.labels[_rep][_k] == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]

			y_upper = y_lower + size_cluster_i

			color = cm.nipy_spectral(float(i) / _k)

			# Make the silhouette blobs
			ax1.fill_betweenx(np.arange(y_lower, y_upper),
								0, ith_cluster_silhouette_values,
								facecolor=color, edgecolor=color, alpha=kwargs["sil_alpha"])

			# For labeling each silhouette blob with their cluster 
			# numbers as the axis labels
			yticks.append(y_lower + 0.5 * size_cluster_i)

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		############### Silhouette plot settings ##############

		# Put vertical line as average silhouette score among samples
		ax1.axvline(x=_silhouette_avg, color=kwargs["avg_sil_color"], linestyle=kwargs["avg_sil_linestyle"], ymax=0.99)

		# Get array of xticks
		xticks = np.arange(kwargs["xmin"], 1.2, 0.2)

		# Set the xaxis labels / ticks
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.set_yticklabels(range(1, _k+1), fontsize=kwargs["silplot_ticklab_fontsize"])

		ax1.tick_params(axis="both", which="major", labelsize=kwargs["silplot_ticklab_fontsize"], colors="black", left=True, bottom=True)

		ax1.set_title("Silhouettes", pad=kwargs["plot_title_pad"])
		ax1.set_xlabel("Silhouette Coefficients", fontsize=kwargs["silplot_xlab_fontsize"])

		ax1.set_ylabel("Cluster Labels", fontsize=kwargs["silplot_ylab_fontsize"])

	def _plot_clusters(self, _k, _rep, kwargs):

		# Check if 3d axes
		if kwargs["axes"] == 3:
			projection = "3d"
		elif kwargs["axes"] == 2:
			projection = None
		else:
			projection = "3d"
			print("Warning: plot_msw=True but axes > 3."
				" Plotting in 3D\n")

		ax2 = plt.subplot(1, 2, 2, projection=projection)
		sns.despine(ax=ax2, offset=5)

		# 2nd Plot showing the actual clusters formed
		colors = cm.nipy_spectral(self.labels[_rep][_k].astype(float) / _k)

		if kwargs["axes"] == 3:
			ax2.scatter(self.coords[_rep][:, 0], self.coords[_rep][:, 1], self.coords[_rep][:, 2], marker=kwargs["marker"], s=kwargs["point_size"], lw=0, alpha=kwargs["point_alpha"], c=colors, edgecolor="k")

			# Labeling the clusters
			centers = self.models[_rep][_k].cluster_centers_

			for i, c in enumerate(centers, start=1):

				x2, y2, _ = proj3d.proj_transform(c[0], c[1], c[2], 
					ax2.get_proj())

				ax2.annotate(
				"{}".format(i), 
				xy = (x2, y2), 
				xytext = (x2, y2),
				textcoords = "data", 
				bbox=dict(boxstyle=kwargs["cluster_lab_shape"], 
					fc=kwargs["cluster_lab_color"], 
					ec="k",
					alpha=kwargs["cluster_lab_alpha"],
					pad=kwargs["cluster_lab_3d_shape_pad"])
				)

		else:
			ax2.scatter(self.coords[_rep][:, 0], self.coords[_rep][:, 1], marker=kwargs["marker"], s=kwargs["point_size"], lw=0, alpha=kwargs["point_alpha"],
					c=colors, edgecolor='k')

			# Labeling the clusters
			centers = self.models[_rep][_k].cluster_centers_

			# Draw white circles at cluster centers
			ax2.scatter(centers[:, 0], centers[:, 1], marker="o",
						c=kwargs["cluster_lab_color"], alpha=kwargs["cluster_lab_alpha"], s=kwargs["cluster_lab_2d_shapesize"], edgecolor="k")

			for i, c in enumerate(centers, start=1):
				ax2.scatter(c[0], c[1], marker="{}".format(i), alpha=kwargs["cluster_lab_alpha"], s=kwargs["cluster_lab_2d_textsize"], edgecolor='k')

		ax2.set_title("Clustered Data", pad=kwargs["plot_title_pad"])
		ax2.set_xlabel("\n{} 1".format(self.method), fontsize=kwargs["cluster_xlab_fontsize"])

		ax2.set_ylabel("\n{} 2".format(self.method), fontsize=kwargs["cluster_ylab_fontsize"])

		if kwargs["axes"] == 3:
			ax2.set_zlabel("\n{} 3".format(self.method), fontsize=kwargs["cluster_zlab_fontsize"])

		ax2.tick_params(axis="both", which="major", labelsize=kwargs["cluster_ticklab_fontsize"], colors="black")

	def msw(self, plot_msw_clusters=False, plot_msw_line=False, plot_all=False, show_all_plots=False, show_clusters_plot=False, show_msw_lineplot=False, axes=2, figwidth=9, figheight=3.5, xmin=0, background_style="white", bottom_margin=None, top_margin=None, left_margin=None, right_margin=None, avg_sil_color="red", avg_sil_linestyle="--", sup_title_y=1.01, plot_title_pad=2, silplot_xlab_fontsize="large", silplot_ylab_fontsize="large", silplot_ticklab_fontsize="large", sil_alpha=0.7, marker="o", cluster_lab_color="#A4D4FF", cluster_lab_shape="circle", cluster_lab_2d_shapesize=400, cluster_lab_2d_textsize=100, cluster_lab_3d_shape_pad=0.3, cluster_xlab_fontsize="large", cluster_ylab_fontsize="large", cluster_zlab_fontsize="large", cluster_ticklab_fontsize="large", cluster_lab_alpha=1.0, point_size=30, point_alpha=0.7, plot_title_fontsize="large"):
		
		if plot_all:
			plot_msw_clusters = True
			plot_msw_line = True

		label_dir = \
			"{}_output/{}/{}/msw/labels".format(self.prefix, self.method, self.clust_method)

		# Makes all directories in path if any don't exist
		Path(label_dir).mkdir(parents=True, exist_ok=True)

		if plot_msw_clusters:
			print("\nUsing MSW to get optimal K...")

			if show_all_plots:
				show_clusters_plot = True
				show_msw_lineplot = True

			plot_dir = \
				"{}_output/{}/{}/msw/plots".format(self.prefix, self.method, self.clust_method)

			# Makes all directories in path if any don't exist
			Path(plot_dir).mkdir(parents=True, exist_ok=True)

		for rep in progressbar(range(self.reps), "Calculating MSW: "):

			if plot_msw_clusters:

				plot_fn = "{}/silhouettes_clusters_{}.pdf".format(plot_dir, rep+1)

				# For multipage PDF plot
				pp = PdfPages(plot_fn)

				# Create a subplot with 1 row and 2 columns
				fig = plt.figure(figsize=(int(figwidth), int(figheight)))

				# Lower the bottom margin so the x-axis label shows
				fig.subplots_adjust(
					bottom=bottom_margin, 
					top=top_margin, 
					left=left_margin, 
					right=right_margin
				)

			silhouettes = dict()
			for k in range(2, self.maxk+1):

				# Compute mean silhouette scores for each K value
				silhouette_avg = \
					silhouette_score(self.coords[rep], self.labels[rep][k])

				silhouettes[k] = silhouette_avg

				if plot_msw_clusters:
					sns.set_style(background_style)

					self._plot_msw(
						silhouette_avg,
						k, 
						pp,
						rep,
						axes=axes, 
						show_clusters_plot=show_clusters_plot,
						figwidth=figwidth, 
						figheight=figheight, 
						xmin=xmin, 
						bottom_margin=bottom_margin, 
						top_margin=top_margin, 
						left_margin=left_margin, 
						right_margin=left_margin, 
						avg_sil_color=avg_sil_color, 
						avg_sil_linestyle=avg_sil_linestyle, 
						sup_title_y=sup_title_y, 
						plot_title_pad=plot_title_pad, 
						silplot_xlab_fontsize=silplot_xlab_fontsize, 
						silplot_ylab_fontsize=silplot_ylab_fontsize, 
						silplot_ticklab_fontsize=silplot_ticklab_fontsize, 
						sil_alpha=sil_alpha, 
						marker=marker, 
						cluster_lab_color=cluster_lab_color, 
						cluster_lab_shape=cluster_lab_shape, 
						cluster_lab_2d_shapesize=cluster_lab_2d_shapesize,
						cluster_lab_2d_textsize=cluster_lab_2d_textsize, 
						cluster_lab_3d_shape_pad=cluster_lab_3d_shape_pad, 
						cluster_xlab_fontsize=cluster_xlab_fontsize, 
						cluster_ylab_fontsize=cluster_ylab_fontsize, 
						cluster_zlab_fontsize=cluster_zlab_fontsize, 
						cluster_ticklab_fontsize=cluster_ticklab_fontsize, 
						cluster_lab_alpha=cluster_lab_alpha, 
						point_size=point_size, 
						point_alpha=point_alpha, 
						plot_title_fontsize=plot_title_fontsize
					)

			if plot_msw_clusters:
				pp.close()
				plt.clf()
				plt.close(fig)

			bestk = max(silhouettes.items(), key=lambda x: x[1])[0]

			if plot_msw_line:
				self._plot_msw_line(silhouettes, bestk, show_msw_lineplot, plot_dir, rep)

			self._write_labels(rep, label_dir, k=bestk)
			self.pred_labels.append(self.labels[rep][bestk])

