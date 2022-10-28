# prototypeftt_AI
AI

S&P 500 Stocks (daily updated)
Stock and company data on all members of the popular financial index.
https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
About this dataset
The Standard and Poor's 500 or S&P 500 is the most famous financial benchmark in the world.

This stock market index tracks the performance of 500 large companies listed on stock exchanges in the United States. As of December 31, 2020, more than $5.4 trillion was invested in assets tied to the performance of this index.

Because the index includes multiple classes of stock of some constituent companies—for example, Alphabet's Class A (GOOGL) and Class C (GOOG)—there are actually 505 stocks in the gauge.
https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks?select=sp500_stocks.csv
Apple (AAPL)
Microsoft (MSFT)
Amazon (AMZN)
Alphabet Class A (GOOGL)
Alphabet Class C (GOOG)


Cryptocoins Historical Prices – CoinGecko
https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrency-historical-prices-coingecko

Bitcoin (BTC)
https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrency-historical-prices-coingecko?select=bitcoin.csv

Ethereum (ETH)
https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrency-historical-prices-coingecko?select=ethereum.csv



Stocks
Initial set up:
1.	Import data from CSV file with date as an index
2.	Clean up the data: remove nans, set index to date, 
3.	Create data frames for individual companies
4.	Create a column called tomorrow and populating it with tomorrow closing price
5.	Set up tomorrow “target” check if tomorrow price is higher than todays price, this will return an integer so we can use it in machine learning

Training an initial machine learning model
1.	Use a RandomForestClassifier because:
a.	It works by training a bunch of individual decision trees with randomized parameters and then averaging the results from those decision trees so because of this process random forests are resistant to overfitting. They can overfit but it is harder for them to overfit than  it is for other models to overfit. They also run relatively quickly and they can pick up non-linear tendencies in the data, so for example the open price is not linearly correlated with the target – for example if the open price is 4000 versus 3000 there is no linear relationship between the open price and the target, if the open price is higher that does not mean that the target will also be higher.   
b.	 Random forest can pick up non-linear relationships which in stock price prediction most of the relationships are non-linear. If you can find a linear relationship then you can make a lot of money
2.	Initialise the model with parameters:
a.	Number of individual decisions trees = 1000, the higher the better accuracy up to a limit
b.	Min sample split – helps us protect against overfitting if you build the tree to deeply, the higher the less accurate the model is 
c.	Random state – a random forest has some randomization built in so setting a random state means that if we run the same model twice the random numbers that will be generated will be in a predictable sequence each time using this random seed of one, so if we re run the model twice we will get the same results, which helps if we to update or improve the model and we want to make sure it is actually the model or the something we did that improved error versus just something random
3.	Split the data into a train and test set 
a.	Put all of the rows except the last 100 rows to the training set
b.	Last 100 rows into the test set
c.	Create a list of the predictors - "Close", "Volume", "Open", "High", "Low"
d.	Train the model using predictors to try to predict the target
e.	Measure how accurate the model is using precision_score what percent the price will go up when we predicted it will go up
4.	Building a backtesting system:
a.	Test the data across multiple years to test how the algorithm is going to handle a lot of different situations to give us more confidence 
b.	Start value – when you backtest your data you want to have a certain amount of data to train your first model, so every trading year has about 250 days, so we are taking 10 years of data and train your first model with the 10 years of data and the step is 250 which means that we will be training a model for about a year and then going to the next year and going to the next year
c.	We will take 10 years of data and predict the values for the 11th year and then we will take the first 11 years of data predict the values for the 12th year and so on, so that we have more confidence in our model (not working only predicts around 750 values)
