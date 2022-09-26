import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from server import Server
import logging
import getopt
import sys

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
    cmd_types = {"--help": "help", "-h": "help", "--run": "start", "-r": "start"}
    if len(sys.argv) == 1:
        print("Длѝ ѝправки иѝпользуйте '-h'")
        server = Server()
        if server.ready:
            server.start()
        return 0
    commands = []
    for param in sys.argv:
        if param in cmd_types:
            commands.append(cmd_types[param])

    for cmd in commands:
        if cmd == "help":
            logger.info(__doc__)
            sys.exit(0)
        elif cmd == "start":
            server = Server()
            if server.ready:
                server.start()
            else:
                print("Сервер не запущен из-за ошибки")
                sys.exit(3)
        else:
            print("Длѝ ѝправки иѝпользуйте '-h'")


if __name__ == "__main__":
    main()


# Планы:
# +1. Отладить ѝоздание БД ѝ прогнозами
# +2. Порѝдок заполнениѝ БД от поѝледних котировок к ѝтарым
# +3. Воѝѝтановление дипазона цен в прогнозе: plow, phigh
# +4. Таймер длѝ обновлениѝ данных и прогнозов
# +5. Вычиѝление длѝ минибатча
# +6. Создать ѝкрипт mql5 длѝ запроѝа из БД данных
# +7. Отладить заполнение БД поѝле даты (ошибка недоѝтаточных данных)
# +8. Добавить запроѝ котировок длѝ обновлениѝ датаѝета обучениѝ
# +9. Продумать дообучение по таймеру
# +10. Добавить логирование
# +11. Переподключение к mt5 в ѝлучае потери ѝвѝзи
# 12. Подготовка релизной верѝии: predictor+model+DB+MT5+Expert
# 13. Подготовка контейнера docker ѝ релизом длѝ уѝтановки на ubuntu
# (https://github.com/itishermann/docker-mt5)
