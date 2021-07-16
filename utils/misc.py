import sys
import os
import functools
import time
import datetime
import platform
import subprocess
import re

from tqdm import tqdm
from tqdm.utils import disp_len, _unicode # for overriding status_print
from numpy.random import choice
#from skopt import BayesSearchCV

def get_indices(l):
	"""
	[Takes a list and returns dict giving indices matching each possible 
	list member]
	Example:
		Input [0, 1, 1, 0, 0]
		Output {0:[0,3,4], 1:[1,2]}
	"""
	ret=dict()
	for member in set(l):
		ret[member] = list()
	i=0
	for el in l:
		ret[el].append(i)
		i+=1
	return(ret)

def all_zero(l):
	"""
	[Returns TRUE if supplied list contains all zeros
	Returns FALSE if list contains ANY non-zero values
	Returns FALSE if list is empty]
	"""
	values=set(l)
	if len(values) > 1:
		return(False)
	elif len(values)==1 and l[0] in [0, 0.0, "0", "0.0"]:
		return(True)
	else:
		return(False)

def weighted_draw(d, num_samples=1):
	choices = list(d.keys())
	weights = list(d.values())
	return(choice(choices, num_samples, p=weights))

def timer(func):
	"""[print the runtime of the decorated function in the format HH:MM:SS]"""
	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		start_time = time.perf_counter()
		value = func(*args, **kwargs)
		end_time = time.perf_counter()
		run_time = end_time - start_time
		final_runtime = str(datetime.timedelta(seconds=run_time))
		print(f"Finshed {func.__name__!r} in {final_runtime}\n")
		return value
	return wrapper_timer

def progressbar(it, prefix="", size=60, file=sys.stdout):
	count = len(it)
	def show(j):
		x = int(size*j/count)
		file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
		file.flush()
	show(0)
	for i, item in enumerate(it):
		yield item
		show(i+1)
	file.write("\n")
	file.flush()

def isnotebook():
	"""[Checks whether in Jupyter notebook]

	Returns:
		[bool]: [True if in Jupyter notebook, False otherwise]
	"""
	try:
		shell = get_ipython().__class__.__name__
		if shell == 'ZMQInteractiveShell':
			# Jupyter notebook or qtconsole
			return True
		elif shell == 'TerminalInteractiveShell':
			# Terminal running IPython
			return False
		else:
			# Other type (?)
			return False  
	except NameError:
		# Probably standard Python interpreter
		return False

def get_processor_name():
	if platform.system() == "Windows":
		return platform.processor()
	elif platform.system() == "Darwin":
		os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
		command ="sysctl -n machdep.cpu.brand_string"
		return subprocess.check_output(command).strip()
	elif platform.system() == "Linux":
		command = "cat /proc/cpuinfo"
		all_info = subprocess.check_output(command, shell=True).strip()
		all_info = all_info.decode("utf-8")
		for line in all_info.split("\n"):
			if "model name" in line:
				return re.sub( ".*model name.*:", "", line,1)
	return ""

class tqdm_linux(tqdm):
	"""
	Decorate an iterable object, returning an iterator which acts exactly
	like the original iterable, but prints a dynamically updating
	progressbar every time a value is requested.
	"""
	@staticmethod
	def status_printer(self, file):
		"""
		Manage the printing and in-place updating of a line of characters.
		Note that if the string is longer than a line, then in-place
		updating may not work (it will print a new line at each refresh).

		Overridden to work with linux HPC clusters. Replaced '\r' with '\n' in fp_write() function.
		"""
		fp = file
		fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover

		def fp_write(s):
			fp.write(_unicode(s))
			fp_flush()

		last_len = [0]

		def print_status(s):
			len_s = disp_len(s)
			fp_write('\n' + s + (' ' * max(last_len[0] - len_s, 0)))
			last_len[0] = len_s

		return print_status

class HiddenPrints:
	"""[Class to supress printing within a with statement]
	"""
	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout

# def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,	n_iter=50, scoring=None, fit_params=None, n_jobs=1,	n_points=1, iid=True, refit=True, cv=None, verbose=0,
# 	pre_dispatch='2*n_jobs', random_state=None,	error_score='raise', 
# 	return_train_score=False
# ):

# 	self.search_spaces = search_spaces
# 	self.n_iter = n_iter
# 	self.n_points = n_points
# 	self.random_state = random_state
# 	self.optimizer_kwargs = optimizer_kwargs
# 	self._check_search_space(self.search_spaces)
# 	self.fit_params = fit_params

# 	super(BayesSearchCV, self).__init__(
# 		estimator=estimator, scoring=scoring,
# 		n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
# 		pre_dispatch=pre_dispatch, error_score=error_score,
# 		return_train_score=return_train_score
# 	)

