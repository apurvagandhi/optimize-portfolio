""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		 
Project Link: https://lucylabs.gatech.edu/ml4t/fall2023/project-2/		 		  		  		    	 		 		   		 		   	   		  		 		  		  		    	 		 		   		 		  
Student Name: Apurva Gandhi	  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: agandhi301		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903862828
		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		    	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		    	   		  		 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data 
import scipy.optimize as spo	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		  		 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  		 		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
    sd=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    ed=dt.datetime(2010, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  		 		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		  		 		  		  		    	 		 		   		 		  
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 		  		  		    	 		 		   		 		  
    statistics.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all_data_frame = get_data(syms, dates)  # automatically adds SPY 
    prices = prices_all_data_frame[syms]  # only portfolio symbols  	
    prices_SPY = prices_all_data_frame["SPY"]  # only SPY, for comparison later  
    allocs = np.asarray(1/len(syms) * np.ones(len(syms)))   
    		  	   		  		 		  		  		    	 		 		   		 		  
    # Computing Stats
    # Daily return for each stock
    def compute_daily_stock_returns(prices):
        daily_returns = prices.copy()
        daily_returns[1:] = (prices[1:] / prices[:-1].values) - 1
        daily_returns.iloc[0, :] = 0 #Set daily returns for row 0 to 0
        return daily_returns
    
    # Daily porfolio return
    def compute_daily_portfolio_return(prices, allocs):
        daily_portfolio_value = calculate_daily_portfolio_value(prices, allocs)
        daily_portfolio_returns = (daily_portfolio_value/daily_portfolio_value.shift(1)) - 1
        daily_portfolio_returns = daily_portfolio_returns.iloc[1:]
        return daily_portfolio_returns
    # Average Daily Portfolio Return
    def calculate_average_daily_portfolio_return(prices, allocs):
        daily_returns = compute_daily_portfolio_return(prices, allocs)    
        return daily_returns.mean()
    
    # Standard Deviation of daily portfolio return
    def calculate_standard_deviation_of_daily_portfolio_return(prices, allocs):
        daily_portfolio_returns = compute_daily_portfolio_return(prices, allocs)
        return daily_portfolio_returns.std()
    
    # Sharpe ratio
    def calculate_sharpe_ratio(prices, allocs):
        return calculate_average_daily_portfolio_return(prices, allocs) / calculate_standard_deviation_of_daily_portfolio_return(prices, allocs)    

    # Total daily porfolio value
    def calculate_daily_portfolio_value(prices, allocs):
        normed = prices / prices.iloc[0, :]        
        alloced = normed * allocs
        return alloced.sum(axis=1)

    # Cumulative return
    def calculate_cumulative_return(prices, allocs):
        daily_portfolio_value = calculate_daily_portfolio_value(prices, allocs)
        return (daily_portfolio_value.iloc[-1]/daily_portfolio_value.iloc[0]) - 1
    
    # Minimum sharpe ratio
    def get_minimum_sharpe_ratio(allocs):
        return -calculate_sharpe_ratio(prices, allocs)
    
    # Calling the minimize function
    bound = []
    for _ in range(len(syms)):
        bound.append((0.0, 1.0))
    result = spo.minimize(get_minimum_sharpe_ratio, np.array([1.0 / len(syms)] * len(syms)), method ='SLSQP', constraints = {'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}, bounds= bound).x   	
    
    # Calculate statistics of the optimized allocated portfolio  		 		  		  		    	 		 		   		 		  
    cr, adr, sddr, sr = [calculate_cumulative_return(prices, result), calculate_average_daily_portfolio_return(prices, result), calculate_standard_deviation_of_daily_portfolio_return(prices, result), calculate_sharpe_ratio(prices, result)] 		  
    
    # Compare daily portfolio value with SPY using a normalized plot
    daily_portfolio_value = calculate_daily_portfolio_value(prices, result)
    normalized_prices_SPY = prices_SPY/prices_SPY.iloc[0]	  		 		  		  		    	 		 		   		 		   		  	   		  		 		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		  		 		  		  		    	 		 		   		 		  
        df_temp = pd.concat([daily_portfolio_value, normalized_prices_SPY], keys=["Portfolio", "SPY"], axis=1)
        ax = df_temp.plot(title = "Daily Portfolio Value and SPY")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.savefig('Figure1.png')	
        plt.clf()  	   		  		 		  		  		    	 		 		   		 		  		  	   		  		 		  		  		    	 		 		   		 		  
      		  		 		  		  		    	 		 		   		 		  
    return result, cr, adr, sddr, sr
               		  	   		  		 		  		  		    	 		 		   		 		  
def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2009, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2010, 1, 1)  		  	   		  		 		  		  		    	 		 		   		 		  
    symbols = ["GOOG", "AAPL", "GLD", "XOM"]  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True  		  	   		  		 		  		  		    	 		 		   		 		  
    )  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  	  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		   		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
