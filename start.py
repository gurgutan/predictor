import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import getopt
import logging
from server import Server

logging.basicConfig(
    handlers=(
        logging.FileHandler("psrv.log", encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-7s | %(message)-120s",
    datefmt="%d.%m.%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) == 1:
        print("Для справки используйте '-h'")
        server = Server()
        if server.ready:
            server.start()
        return 0
    for param in sys.argv:
        if param in ("-h", "--help"):
            logger.info(__doc__)
            sys.exit(0)
        if param in ("-r", "--run"):
            server = Server()
            if server.ready:
                server.start()
            else:
                print("Сервер не запущен из-за ошибки")
                sys.exit(3)


if __name__ == "__main__":
    main()


# Планы:
# +1. Отладить создание БД с прогнозами
# +2. Порядок заполнения БД от последних котировок к старым
# +3. Восстановление дипазона цен в прогнозе: plow, phigh
# +4. Таймер для обновления данных и прогнозов
# +5. Вычисление для минибатча
# +6. Создать скрипт mql5 для запроса из БД данных
# +7. Отладить заполнение БД после даты (ошибка недостаточных данных)
# 8. Добавить запрос котировок для обновления датасета обучения
# 9. Продумать дообучение по таймеру
# +10. Добавить логирование
# +11. Переподключение к mt5 в случае потери связи
