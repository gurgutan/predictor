import sys
import getopt
from server import Server


def main():
    if len(sys.argv) == 1:
        print("Для справки используйте '-h'")
        return 0
    for param in sys.argv:
        if param in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        if param in ("-r", "--run"):
            server = Server()
            if server.ready:
                print("Сервер запущен")
                server.start()
            else:
                print("Сервер не запущен из-за ошибки")
                sys.exit(3)


if __name__ == "__main__":
    main()


# Планы:
# +1. Отладить создание БД с прогнозами
# 2. Порядок заполнения БД от последних котировок к старым
# 3. Восстановление дипазона цен в прогнозе: plow, phigh
# 4. Таймер для обновления данных и прогнозов
# 5. Вычисление для минибатча
# 6. Создать скрипт mql5 для запроса из БД данных
