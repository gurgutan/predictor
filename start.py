import sys
import getopt
from server import Server


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error as msg:
        print(msg)
        print("Для справки используйте --help")
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        if o in ("-r", "--run"):
            server = Server("/db/predictions.sqlite")
            if server.ready:
                print("Сервер запущен")
                server.start()
            else:
                print("Сервер не запущен из-за ошибки")
                sys.exit(3)


if __name__ == "__main__":
    main()
