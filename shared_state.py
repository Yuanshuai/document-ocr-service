import threading


class _SharedState:
    def __init__(self):
        self.gpu_lock = threading.Lock()
        self.structure_lock = threading.Lock()
        self.handwriting_lock = threading.Lock()


shared_state = _SharedState()
