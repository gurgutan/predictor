import MetaTrader5 as mt5
import pytz

MAGIC = 197979


def SendOrder(symbol, volume, tp=512, sl=512, comment=""):
    pass
    # подготовим структуру запроса для покупки
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, f"Символ '{symbol}' не найден")
        return False
    # если символ недоступен в MarketWatch, добавим его
    if not symbol_info.visible:
        print(symbol, "Символ не включен, включаю...")
        if not mt5.symbol_select(symbol, True):
            print(f"ошибка выполнения symbol_select({symbol})")
            return False
    point = mt5.symbol_info(symbol).point
    deviation = 20
    if volume > 0:
        lot = round(volume, 2)
        price = mt5.symbol_info_tick(symbol).ask
        order_type = mt5.ORDER_TYPE_BUY
        stop_loss = price - sl * point
        take_profit = price + tp * point
    elif volume < 0:
        lot = round(-volume, 2)
        price = mt5.symbol_info_tick(symbol).bid
        order_type = mt5.ORDER_TYPE_SELL
        stop_loss = price + sl * point
        take_profit = price - tp * point
    else:
        print("Ошибка в volume")
        return False
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": deviation,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    # проверим результат выполнения
    print(f"order_send(): {symbol} {lot} по {price}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"order_send failed, retcode={result.retcode}")
        # запросим результат в виде словаря и выведем поэлементно
        result_dict = result._asdict()
        for field in result_dict.keys():
            print(f"   {field}={result_dict[field]}")
            # если это структура торгового запроса, то выведем её тоже поэлементно
            if field == "request":
                traderequest_dict = result_dict[field]._asdict()
                for tradereq_filed in traderequest_dict:
                    print(
                        f"  traderequest: {tradereq_filed}={ traderequest_dict[tradereq_filed]}"
                    )
        return False
    return True

