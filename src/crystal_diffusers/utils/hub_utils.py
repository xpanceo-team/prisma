from contextlib import contextmanager

import datasets


@contextmanager
def no_datasets_progress_bar():
    """
    Temporarily disable `datasets` progress bars.
    On exit, restore to whatever state it was before.
    """
    was_enabled = datasets.is_progress_bar_enabled()
    datasets.logging.disable_progress_bar()
    try:
        yield
    finally:
        if was_enabled:
            datasets.logging.enable_progress_bar()


