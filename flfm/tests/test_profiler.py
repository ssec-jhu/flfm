import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from flfm.profiler import ProfiledProcess, Profiler


@pytest.fixture
def python_buffer():
    python_buffer = os.environ["PYTHONUNBUFFERED"]
    os.environ["PYTHONUNBUFFERED"] = "1"
    yield
    os.environ["PYTHONUNBUFFERED"] = python_buffer


def random_number_cpu_consumer(n, verbose=False):
    import random

    for i in range(int(n)):
        x = random.randint(0, 100)
        if verbose:
            print(x)


class TestProfiler:
    def test_random_number_generator(self, python_buffer):
        p = Profiler()
        fast_data = p.profile(
            random_number_cpu_consumer, args=[1e8], kwargs={"verbose": False}, verbose=False, interval=0.5, plot=False
        )
        slow_data = p.profile(
            random_number_cpu_consumer, args=[1e6], kwargs={"verbose": True}, verbose=False, interval=0.5, plot=False
        )

        assert np.mean([x[1] for x in slow_data]) < 90  # Having to print to IO will prevent full CPU usage.
        assert np.mean([x[1] for x in fast_data]) > 90

    def test_save(self, tmpdir):
        p = Profiler()
        data = p.profile(
            random_number_cpu_consumer, args=[1e3], kwargs={"verbose": False}, verbose=False, interval=0.5, plot=False
        )

        filename = Path(tmpdir) / "data.csv"
        assert not filename.exists()
        p.save(data, filename)
        assert filename.exists()

    def test_trim(self):
        p = Profiler()
        sleep = ProfiledProcess.SLEEP * int(1e9)
        data = [
            (0, 0, 0),
            (sleep - 1, 0, 0),
            (sleep + 1, 0, 0),
        ]
        trimmed_data = p.trim_data(data)
        assert len(trimmed_data) == 1
        assert trimmed_data == [
            (sleep + 1, 0, 0),
        ]

    def test_plot(self):
        p = Profiler()
        data = p.profile(
            random_number_cpu_consumer, args=[1e3], kwargs={"verbose": False}, verbose=False, interval=0.5, plot=False
        )
        figure = p.plot(data)
        assert isinstance(figure, plt.Figure)
