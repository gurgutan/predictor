import sys
import getopt
from server import Server


def main():
    if(len(sys.argv) == 1):
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
