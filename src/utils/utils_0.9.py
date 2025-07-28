import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np

def Write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
    for i in net_worth: 
        Date += " {}".format(i)
    #print(Date)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
    file.write(Date+"\n")
    file.close()

class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, Render_range, Show_reward=False, Show_indicators=False):
        self.Volume = deque(maxlen=Render_range)
        self.net_worth = deque(maxlen=Render_range)
        self.render_data = deque(maxlen=Render_range)
        self.Render_range = Render_range
        self.Show_reward = Show_reward
        self.Show_indicators = Show_indicators

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8)) 

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((8,1), (0,0), rowspan=5, colspan=1)
        
        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((8,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)
        self.ax5 = plt.subplot2grid((8,1), (6,0), rowspan=1, colspan=1, sharex=self.ax1)
        self.ax6 = plt.subplot2grid((8,1), (7,0), rowspan=1, colspan=1, sharex=self.ax1)
        
        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
        
        # Add paddings to make graph easier to view
        #plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

        # define if show indicators
        if self.Show_indicators:
            self.Create_indicators_lists()

    def Create_indicators_lists(self):
        # Create a new axis for indicatorswhich shares its x-axis with volume
        self.ax4 = self.ax2.twinx()
        
        self.ema5 = deque(maxlen=self.Render_range)
        self.ema8 = deque(maxlen=self.Render_range)
        self.ema13 = deque(maxlen=self.Render_range)

        self.MACD = deque(maxlen=self.Render_range)
        self.MACD_signal = deque(maxlen=self.Render_range)
        self.MACD_hist = deque(maxlen=self.Render_range)
        
        self.RSI = deque(maxlen=self.Render_range)


    def Plot_indicators(self, df, Date_Render_range):
        self.ema5.append(df["ema5"])
        self.ema8.append(df["ema8"])
        self.ema13.append(df["ema13"])

        self.MACD.append(df["MACD"])
        self.MACD_signal.append(df["MACD signal"])
        self.MACD_hist.append(df["MACD hist"])
        self.RSI.append(df["RSI"])

        # Add Simple Moving Average
        #self.ax1.plot(Date_Render_range, self.ema5,'r-')
        #self.ax1.plot(Date_Render_range, self.ema8,'g-')
        #self.ax1.plot(Date_Render_range, self.ema13,'b-')

        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        # # Add Moving Average Convergence Divergence
        self.ax5.plot(Date_Render_range, self.MACD,'r-')
        self.ax5.plot(Date_Render_range, self.MACD_signal,'b-')
        self.ax5.plot(Date_Render_range, self.MACD_hist,'g-')
        self.ax5.fill_between(Date_Render_range, 0, self.MACD_hist, where=(np.array(self.MACD_hist)-1) < -1 , color='red')
        self.ax5.fill_between(Date_Render_range, 0, self.MACD_hist, where=(np.array(self.MACD_hist)-1) > -1 , color='green')

        # # Add Relative Strength Index
        self.ax6.plot(Date_Render_range, self.RSI,'b-')

    # Render the environment to the screen
    #def render(self, Date, Open, High, Low, Close, Volume, net_worth, trades):
    def render(self, df, net_worth, trades):
        Date = df["time"]
        Open = df["open"]
        High = df["high"]
        Low = df["low"]
        Close = df["close"]
        Volume = df["tick_volume"]
        # append volume and net_worth to deque list
        self.Volume.append(Volume)
        self.net_worth.append(net_worth)

        # before appending to deque list, need to convert Date to special format
        Date = mpl_dates.date2num([pd.to_datetime(Date)])[0]
        self.render_data.append([Date, Open, High, Low, Close])
        
        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8/24, colorup='green', colordown='red', alpha=0.8)

        # Put all dates to one list and fill ax2 sublot with volume
        Date_Render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(Date_Render_range, self.Volume, 0, color = 'gray')

        if self.Show_indicators:
            self.Plot_indicators(df, Date_Render_range)

        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(Date_Render_range, self.net_worth, color="black")
        
        # beautify the x-labels (Our Date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        minimum = np.min(np.array(self.render_data)[:,1:])
        maximum = np.max(np.array(self.render_data)[:,1:])
        RANGE = maximum - minimum


        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['datetime'])])[0]
            if trade_date in Date_Render_range:
                if trade['type'] == 'buy':
                    high_low = trade['low'] - RANGE*0.02
                    ycoords = trade['low'] - RANGE*0.08
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s = 120, edgecolors='none', marker="^")
                else:
                    high_low = trade['high'] + RANGE*0.02
                    ycoords = trade['high'] + RANGE*0.06
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 120, edgecolors='none', marker="v")

                if self.Show_reward:
                    try:
                        self.ax1.annotate('{0:.2f}'.format(trade['Reward']), (trade_date-0.02, high_low), xytext=(trade_date-0.02, ycoords),
                                                   bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
                    except:
                        pass

        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        #plt.show(block=False)
        # Necessary to view frames before they are unrendered
        #plt.pause(0.001)

        """Display image with OpenCV - no interruption"""

        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with OpenCV or any operation you like
        cv2.imshow("Bitcoin trading bot",image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        else:
            return img
        

def Plot_OHCL(df):
    df_original = df.copy()
    # necessary convert to datetime
    df["datetime"] = pd.to_datetime(df.Date)
    df["datetime"] = df["date"].apply(mpl_dates.date2num)

    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # We are using the style ‘ggplot’
    plt.style.use('ggplot')
    
    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16,8)) 

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax5 = plt.subplot2grid((8,1), (6,0), rowspan=1, colspan=1, sharex=ax1)
    ax6 = plt.subplot2grid((8,1), (7,0), rowspan=1, colspan=1, sharex=ax1)

    candlestick_ohlc(ax1, df.values, width=0.8/24, colorup='green', colordown='red', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # Add Simple Moving Average
    ax1.plot(df["Date"], df_original['ema5'],'-')
    ax1.plot(df["Date"], df_original['ema8'],'-')
    ax1.plot(df["Date"], df_original['ema13'],'-')

    # # Add Moving Average Convergence Divergence
    ax5.plot(df["Date"], df_original['MACD'],'-')
    ax5.plot(df["Date"], df_original['MACD signal'],'-')
    ax5.fill_between(df["Date"], 0,df_original['MACD'], where=(df_original['MACD']-1) < -1 , color='red')
    ax5.fill_between(df["Date"], 0,df_original['MACD'], where=(df_original['MACD']-1) > -1 , color='green')

    # # Add Relative Strength Index
    ax6.plot(df["Date"], df_original['RSI'],'-')

    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))# %H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    
    plt.show()