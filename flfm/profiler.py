import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import psutil
from multiprocess import get_context

CONTEXT_METHOD = "spawn"
CONTEXT = get_context(CONTEXT_METHOD)


class ProfiledProcess(CONTEXT.Process):
    SLEEP = 2

    def run(self):
        import time

        time.sleep(self.SLEEP)  # Give the parent process time to start profiling.
        super().run()


class Profiler:
    @staticmethod
    def trim_data(data) -> List:
        """Trim data to remove initial sleep period."""
        sleep = ProfiledProcess.SLEEP * int(1e9)  # Convert s -> ns.

        start_time = data[0][0]
        t = [(x[0] - start_time) for x in data]

        # Find start of data, i.e., that after initial sleep.
        for i, v in enumerate(t):
            if v >= sleep:
                return data[i:]

        return data

    @staticmethod
    def plot(data: List, show: bool = False) -> plt.Figure:
        """Plot and/or return figure of CPU% and memory usage as a function of elapsed time."""

        t = [x[0] / 1e9 for x in data]  # Note: time has unit ns.
        cpu = [x[1] for x in data]
        memory = [x[2] / 1e9 for x in data]  # Note: memory has unit bytes.

        # Plot CPU utiliaztion.
        fig, ax1 = plt.subplots()
        color = "tab:red"
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("CPU utilization (%)", color=color)
        ax1.plot(t, cpu, color=color)
        mean_cpu = np.mean(cpu)
        ax1.axhline(y=mean_cpu, color=color, linestyle="-", label=f"Mean CPU: {mean_cpu:.2f}%")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid()

        # Plot memory usage.
        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Memory (GB)", color=color)
        ax2.plot(t, memory, color=color)
        max_memory = np.max(memory)
        ax2.axhline(y=max_memory, color=color, linestyle="-", label=f"Max: {max_memory:.2f}GB")
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        fig.legend()
        if show:
            plt.show()

        return fig

    @staticmethod
    def save(data, filename):
        """Save data as CSV to filename."""
        np.savetxt(filename, data, delimiter=",")

    def profile(self, func, args=None, kwargs=None, output_filename=None, interval=1, verbose=False, plot=False):
        """Profile CPU and memory utilization."""
        process = ProfiledProcess(target=func, args=args or [], kwargs=kwargs or {})
        data = []

        t0 = time.time()
        process.start()
        pid = process.pid
        profiler = psutil.Process(pid)

        # Initial profile call. See https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_percent.
        profiler.cpu_percent(interval=interval)

        while process.is_alive():
            try:
                # cpu_percent() is non-blocking for interval=None. Do this and explicitly sleep, to reduce the
                # overhead of comparing clock times by cpu_percent() ib blocking. mode.
                cpu_percent = profiler.cpu_percent(interval=None)
                memory = profiler.memory_info()
            except psutil.NoSuchProcess:
                break
            data.append((time.perf_counter_ns(), cpu_percent, memory.rss))
            if verbose:
                print(f"CPU: {cpu_percent}%, MEM: {memory.rss / 1e6:.2f}MB")
            time.sleep(interval)

        process.join()
        runtime = time.time() - t0 - ProfiledProcess.SLEEP
        print(f"The approximate end-to-end runtime: {runtime:.2f}")

        data = self.__class__.trim_data(data)

        if output_filename:
            self.__class__.save(data, output_filename)

        if process.exitcode:
            raise RuntimeError(f"Profiled func failed with exitcode: {process.exitcode}")

        if plot:
            self.__class__.plot(data)

        return data
