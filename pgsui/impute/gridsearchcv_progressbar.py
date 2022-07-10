import time
import sys
import warnings

from sklearn import model_selection
from joblib import Parallel


try:
    from ..utils.misc import isnotebook
except:
    from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class _FakeParallel(object):
    """Useless decorator that is used like Parallel"""

    def __init__(self, n_jobs=1, *args, **kwargs):
        if n_jobs != 1:
            warnings.warn(
                "You have set n_jobs=%s to use parallel processing, "
                "but it has currently no effect. Please install joblib or "
                "scikit-learn to use it." % (n_jobs,)
            )

    def __call__(self, iterable):

        return list(iterable)


def _fake_delayed(func):
    return func


# try to import from joblib, otherwise, create a dummy function with no effect
try:
    from joblib import Parallel, delayed
except ImportError:
    try:
        from sklearn.externals.joblib import Parallel, delayed
    except ImportError:
        Parallel = _FakeParallel
        delayed = _fake_delayed

# copy from MNE-Python
class ProgressBar:
    """Class for generating a command-line progressbar
    Parameters
    ----------
    max_value : int
        Maximum value of process (e.g. number of samples to process, bytes to
        download, etc.).
    initial_value : int
        Initial value of process, useful when resuming process from a specific
        value, defaults to 0.
    title : str
        Message to include at end of progress bar.
    max_chars : int
        Number of characters to use for progress bar (be sure to save some room
        for the message and % complete as well).
    progress_character : char
        Character in the progress bar that indicates the portion completed.
    spinner : bool
        Show a spinner.  Useful for long-running processes that may not
        increment the progress bar very often.  This provides the user with
        feedback that the progress has not stalled.
    """

    spinner_symbols = ["|", "/", "-", "\\"]
    template = "\r[{0}{1}] {2:.0f}% {3} {4:.02f} sec | {5} "

    def __init__(
        self,
        title="",
        max_value=1,
        initial_value=0,
        max_chars=40,
        progress_character=".",
        spinner=False,
        verbose_bool=True,
    ):
        self.cur_value = initial_value
        self.max_value = float(max_value)
        self.title = title
        self.max_chars = max_chars
        self.progress_character = progress_character
        self.spinner = spinner
        self.spinner_index = 0
        self.n_spinner = len(self.spinner_symbols)
        self._do_print = verbose_bool
        self.start = time.time()

        self.closed = False
        self.update(initial_value)

    def update(self, cur_value, title=None):
        """Update progressbar with current value of process
        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            (cur_value / max_value) * 100
        title : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        # Ensure floating-point division so we can get fractions of a percent
        # for the progressbar.
        self.cur_value = cur_value
        progress = min(float(self.cur_value) / self.max_value, 1.0)
        num_chars = int(progress * self.max_chars)
        num_left = self.max_chars - num_chars

        # Update the message
        if title is not None:
            self.title = title

        # time from start
        duration = time.time() - self.start

        # The \r tells the cursor to return to the beginning of the line rather
        # than starting a new line.  This allows us to have a progressbar-style
        # display in the console window.
        bar = self.template.format(
            self.progress_character * num_chars,
            " " * num_left,
            progress * 100,
            self.spinner_symbols[self.spinner_index],
            duration,
            self.title,
        )
        # Force a flush because sometimes when using bash scripts and pipes,
        # the output is not printed until after the program exits.
        if self._do_print:
            sys.stdout.write(bar)
            sys.stdout.flush()
        # Increament the spinner
        if self.spinner:
            self.spinner_index = (self.spinner_index + 1) % self.n_spinner

        if progress == 1:
            self.close()

    def update_with_increment_value(self, increment_value, title=None):
        """Update progressbar with the value of the increment instead of the
        current value of process as in update()
        Parameters
        ----------
        increment_value : int
            Value of the increment of process.  The percent of the progressbar
            will be computed as
            (self.cur_value + increment_value / max_value) * 100
        title : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        self.cur_value += increment_value
        self.update(self.cur_value, title)

    def close(self):
        if not self.closed:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.closed = True

    def __call__(self, sequence):
        sequence = iter(sequence)
        while True:
            try:
                yield next(sequence)
                self.update_with_increment_value(1)
            except StopIteration:
                return


class GridSearchCVProgressBar(model_selection.GridSearchCV):
    """Monkey patch Parallel to have a progress bar during grid search"""

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""

        iterator = super(GridSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)

        cv = model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, "n_splits", 3)
        max_value = n_candidates * n_splits
        print("test")

        class ParallelProgressBar(Parallel):
            def __call__(self, iterable):
                bar = ProgressBar(max_value=max_value, title="GridSearchCV")
                iterable = bar(iterable)
                return super(ParallelProgressBar, self).__call__(iterable)

        # Monkey patch
        model_selection._search.Parallel = ParallelProgressBar

        return iterator
