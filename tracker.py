import time

class Tracker:
    data = {}
    def start(name):
        if name not in Tracker.data:
            Tracker.data[name] = [0.0, 0.0, 0]
        Tracker.data[name][2] += 1
        Tracker.data[name][1] = time.time()
    def stop(name):
        total, start, count = Tracker.data[name]
        Tracker.data[name][0] = total + (time.time() - start)
    def show():
        for name in Tracker.data:
            total, start, count = Tracker.data[name]
            avg = total / count if count > 0 else 0
            print(f'{name}: {total}s @ {avg}/s - {count} times')