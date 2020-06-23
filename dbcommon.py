import sqlite3
import datetime as dt


def db_open(dbname):
    try:
        db = sqlite3.connect(dbname)
        cursor = db.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS pdata("
            + "rdate integer PRIMARY KEY, "
            + "rprice real, "
            + "pmodel integer, "
            + "pdate integer, "
            + "plow real, "
            + "phigh real, "
            + "prob real)"
        )
        db.commit()
        return db
    except Exception as e:
        print("Ошибка открытия БД '%s':" % dbname)
        return None


def db_replace(db, data):
    cursor = db.cursor()
    cursor.executemany(
        "INSERT OR REPLACE INTO pdata("
        + "rdate integer, "
        + "rprice real, "
        + "pmodel text, "
        + "pdate integer, "
        + "plow real, "
        + "phigh real, "
        + "prob real) VALUES(?,?,?,?,?,?,?)",
        data)
    db.commit()


def db_update_real_prices(db, data):
    """
    db - connection
    data - (rdate integer, rprice real)
    """
    cursor = db.cursor()
    cursor.executemany(
        "INSERT OR REPLACE INTO pdata("
        + "rdate integer, "
        + "rprice real) VALUES(?,?)",
        data)
    db.commit()


def db_select(db, from_date, to_date):
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM pdata WHERE rdate BETWEEN ? and ?",
        (from_date, to_date)
    )
    return cursor.fetchall()


def db_get_lowdate(db):
    cursor = db.cursor()
    cursor.execute("select max(rdate) from pdata")
    return cursor.fetchone()
