from datetime import timedelta, datetime


class DelayTimer:
    def __init__(self, seconds=60, shift=0, name="timer"):
        """Параметры: seconds - задержка таймера в секундах, shift - сдвиг таймера от начала часа"""
        self.seconds = seconds
        # Выравнивание времени по начальной дате
        self.last = datetime(2000, 1, 1) + timedelta(seconds=shift)
        self.name = name

    def __repr__():
        return f"{self.name}: {self.remained()}"

    def elapsed(self):
        """
        Возвращает True, если заданный интервал времени уже прошел и False - в противном случае
        """
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
