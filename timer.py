from datetime import timedelta, datetime


class DelayTimer:
    def __init__(self, seconds=60):
        self.seconds = seconds
        self.last = datetime.now()

    def elapsed(self):
        delta = (datetime.now() - self.last).total_seconds()
        if delta > self.seconds:
            self.last = datetime.now() - timedelta(seconds=delta % self.seconds)
            return True
        else:
            return False


# Тест
# t = DelayTimer(5)
# while True:
#     sleep(3)
#     if t.check():
#         print(datetime.now())
