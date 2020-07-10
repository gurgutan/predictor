import sqlite3
import datetime as dt


def db_open(dbname):
    model_info_ddl = f"CREATE TABLE IF NOT EXISTS model_info(
        name TEXT NOT NULL PRIMARY KEY,
        input_shape STRING,
        output_shape STRING,
        future INTEGER NOT NULL,
        timeunit INTEGER NOT NULL);"
    pdata_ddl = f"CREATE TABLE IF NOT EXISTS pdata(
        rdate      INTEGER PRIMARY KEY,
        rprice     REAL    NOT NULL,
        symbol     TEXT,
        model      TEXT    NOT NULL,
        pdate      INTEGER NOT NULL,
        plow       REAL    NOT NULL,
        phigh      REAL    NOT NULL,
        confidence REAL    NOT NULL);"
    symbol_info_ddl = f"CREATE TABLE IF NOT EXISTS symbol_info(
        name        TEXT    PRIMARY KEY,
        timeframe   INTEGER NOT NULL,
        volume_min  REAL    NOT NULL,
        volume_max  REAL    NOT NULL,
        volume_step REAL    NOT NULL);"
    pdata_idx_ddl = f"CREATE INDEX pdate_idx ON pdata(pdate DESC);"
    confidence_idx_ddl = f"CREATE INDEX confidence_idx ON pdata(confidence DESC);"

    try:
        db = sqlite3.connect(dbname)
        cursor = db.cursor()
        cursor.execute(pdata_ddl)
        cursor.execute(model_info_ddl)
        cursor.execute(symbol_info_ddl)
        cursor.execute(pdata_idx_ddl)
        cursor.execute(confidence_idx_ddl)
        db.commit()
        return db
    except Exception as e:
        print("Ошибка открытия БД '%s':" % dbname)
        return None


def db_replace(db, data):
    if data is None:
        return False
    cursor = db.cursor()
    cursor.executemany(
        "INSERT OR REPLACE INTO pdata("
        + "rdate, "
        + "rprice, "
        + "symbol, "
        + "model, "
        + "pdate, "
        + "plow, "
        + "phigh, "
        + "confidence) VALUES(?,?,?,?,?,?,?,?)",
        data,
    )
    db.commit()
    return True


def db_update_real_prices(db, data):
    """
    db - connection
    data - (rdate integer, rprice real)
    """
    cursor = db.cursor()
    cursor.executemany(
        "INSERT OR REPLACE INTO pdata(rdate, rprice) VALUES(?,?)", data,
    )
    db.commit()


def db_select(db, from_date, to_date):
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM pdata WHERE rdate BETWEEN ? and ?", (from_date, to_date)
    )
    return cursor.fetchall()


def db_get_lowdate(db):
    cursor = db.cursor()
    cursor.execute("select max(rdate) from pdata")
    value = cursor.fetchone()[0]
    if value is None:
        return None
    result = dt.datetime.fromtimestamp(int(value))
    return result
