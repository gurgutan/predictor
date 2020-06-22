import json
import sqlite3


# 1. Инициализация БД
# 2. Инициализация mt5
# 3. Запрос котировок от mt5
# 4. Построение прогноза и запись в БД


class Server(object):
    def __init__(self):
        self.ready = False
        with open("config.json") as config_file:
            data = json.load(config_file)
            self.dbname = data["dbname"]
            self.daterange = data["daterange"]
        if self.__initdb__() and self.__initmt5__() and self.__initpredictor__():
            print("Сервер инициализирован")
            self.ready = True
        else:
            print("Ошибка инициализации сервера")
            self.ready = False

    def __initdb__(self):
        try:
            self.con = sqlite3.connect(self.dbname)
            self.cursor = self.con.cursor()
            self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS pdata("
                + "id integer PRIMARY KEY, "
                + "rdate integer, "
                + "rprice real, "
                + "pmodel integer, "
                + "pdate integer, "
                + "plow real, "
                + "phigh real, "
                + "prob real)"
            )
            self.con.commit()
            return True
        except Exception as e:
            print("Ошибка открытия БД '%s':" % self.dbname)
            return False
        finally:
            self.con.close()
        return False

    def __initmt5__(self):
        return True

    def __initpredictor__(self):
        return True

    def start(self):
        while True:
            self.request_data()
            self.compute()

