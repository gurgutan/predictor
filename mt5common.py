import MetaTrader5 as mt5
import pytz
import logging


MAGIC = 197979
logger = logging.getLogger(__name__)


def init_mt5(mt5path):
    if not mt5.initialize(path=mt5path):
        logger.error("Ошибка подключения к терминалу MT5")
        mt5.shutdown()
        return False
    logger.info("Подключено к терминалу '%s' версия %s" % (mt5path, str(mt5.version())))
    # logger.info(get_account_info())
    return True


def is_mt5_connected():
    info = mt5.account_info()
    if mt5.last_error()[0] < 0:
        return False
    return True


def get_account_info():
    account_info = mt5.account_info()
    if account_info != None:
        return str(account_info)
    else:
        return ""


def get_equity():
    account_info = mt5.account_info()
    if account_info != None:
        return int(account_info._asdict()["equity"])
    else:
        return None


def is_trade_allowed():
    terminal_info = mt5.terminal_info()._asdict()
    if terminal_info == None:
        logger.error(f"Ошибка запроса terminal_info()")
        return False
    return terminal_info["trade_allowed"] and (not terminal_info["tradeapi_disabled"])


def send_order(symbol, volume, tp=512, sl=512, comment=""):
    # подготовим структуру запроса для покупки
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Символ '{symbol}' не найден")
        return False
    # если символ недоступен в MarketWatch, добавим его
    if not symbol_info.visible:
        logger.info(f"Символ {symbol} не включен, включаю...")
        if not mt5.symbol_select(symbol, True):
            logger.error(f"ошибка выполнения symbol_select({symbol})")
            return False
    point = mt5.symbol_info(symbol).point
    deviation = 20
    if volume > 0:
        lot = round(volume, 2)
        price = mt5.symbol_info_tick(symbol).ask
        order_type = mt5.ORDER_TYPE_BUY
        stop_loss = round(price - sl * point, 5)
        take_profit = round(price + tp * point, 5)
        str_order_type = "buy"
    elif volume < 0:
        lot = round(-volume, 2)
        price = mt5.symbol_info_tick(symbol).bid
        order_type = mt5.ORDER_TYPE_SELL
        stop_loss = round(price + sl * point, 5)
        take_profit = round(price - tp * point, 5)
        str_order_type = "sell"
    else:
        logger.error(f"Ошибка в volume:{volume}")
        return False
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": deviation,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,  # ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    # проверим результат выполнения
    logger.info(
        f"order_send: {symbol}, {str_order_type} {lot} по цене {round(price,5)}"
    )
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Ошибка: order_send, retcode={result.retcode}")
        # запросим результат в виде словаря и выведем поэлементно
        result_dict = result._asdict()
        for field in result_dict.keys():
            logger.error(f"   {field}={result_dict[field]}")
            # если это структура торгового запроса, то выведем её тоже поэлементно
            if field == "request":
                traderequest_dict = result_dict[field]._asdict()
                for tradereq_filed in traderequest_dict:
                    logger.error(
                        f"  traderequest: {tradereq_filed}={traderequest_dict[tradereq_filed]}"
                    )
        return False
    return True

