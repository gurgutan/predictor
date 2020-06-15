from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np

import strategy

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(strategy.TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath,
                            'datas/EURUSD_M5_200001030000_202006122350.csv')
    data = btfeeds.GenericCSVData(dataname=datapath,
                                  fromdate=datetime.datetime(200, 1, 1),
                                  todate=datetime.datetime(2000, 1, 31),
                                  nullvalue=0.0,
                                  separator='\t',
                                  dtformat=('%Y.%m.%d'),
                                  tmformat=('%H:%M:%S'),
                                  time=1,
                                  open=2,
                                  high=3,
                                  low=4,
                                  close=5,
                                  volume=7,
                                  openinterest=-1)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.4f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.4f' % cerebro.broker.getvalue())