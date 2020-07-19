from datetime import timedelta, datetime


class DelayTimer:
    def __init__(self, seconds=60):
        self.seconds = seconds
        # Выравнивание времени по начальной дате
        self.last = datetime(2000, 1, 1)

    def elapsed(self):
        delta = (datetime.now() - self.last).total_seconds()
        if delta > self.seconds:
            self.last = datetime.now() - timedelta(seconds=delta % self.seconds)
            return True
        else:
            return False

    def remained(self):
        """
        Возвращает число секунд, оставшихся до окончания цикла
        """
        delta = (datetime.now() - self.last).total_seconds()
        return delta % self.seconds


# Тест
# t = DelayTimer(5)
# while True:
#     sleep(3)
#     if t.check():
#         print(datetime.now())
