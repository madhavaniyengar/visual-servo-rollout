from threading import Event, Thread

class KeyboardFlags:

    def __init__(self):
        self._step_flag = Event()
        self._quit_flag = Event()
        Thread(target=self._dispatch, daemon=True).start()

    def _dispatch(self):
        while not self._quit_flag.is_set():
            if input().strip() == "q":
                self._quit_flag.set()
            else:
                self._step_flag.set()

    def clear_step_flag(self):
        self._step_flag.clear()

    def quit_flag(self):
        return self._quit_flag.is_set()

    def step_flag(self):
        return self._step_flag.is_set()
