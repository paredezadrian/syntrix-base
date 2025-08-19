import contextlib
import time


@contextlib.contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[timer] {name}: {dt:.3f}s")


class Throughput:
    def __init__(self):
        self.t0 = time.time()
        self.count = 0

    def update(self, n: int):
        self.count += n

    def per_second(self) -> float:
        dt = max(1e-6, time.time() - self.t0)
        return self.count / dt
