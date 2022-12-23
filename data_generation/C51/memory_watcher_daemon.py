import threading
from time import sleep
import psutil
from collections import deque


class MemoryWatcher:
    def __init__(
        self, polling_rate=0.5, max_length=10_000, oom_threshold=1_000_000_000
    ):
        self.max_length = max_length
        self._daemon_run_permision = True
        self.oom_threshold = oom_threshold
        self.queue: deque[int] = deque([], self.max_length)
        self.polling_rate = polling_rate

        self.daemon = threading.Thread(target=self._memory_watcher, daemon=True)
        self.daemon.start()

        print("started memory watcher_daemon")
        print(
            f"daemon will save virtual memory data for {polling_rate * max_length} seconds"
        )

    def _memory_watcher(self):
        while self._daemon_run_permision:
            self.queue.append(psutil.virtual_memory().available)
            sleep(self.polling_rate)

    def is_out_of_memory(self):
        return [ellement < self.oom_threshold for ellement in self.queue]

    def stop(self):
        self._daemon_run_permision = False

    def restart(self):
        if self._daemon_run_permision:
            self.stop()

        self._daemon_run_permision = True

        self.daemon = threading.Thread(target=self._memory_watcher, daemon=True)
        self.daemon.start()

        print("restarted daemon")

    def clear_queue(self):
        self.queue: deque[int] = deque([], self.max_length)
