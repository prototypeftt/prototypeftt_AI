from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

api = KaggleApi()
api.authenticate()

api.dataset_download_files('andrewmvd/sp-500-stocks', path='stockDatasets', unzip=True)
api.dataset_download_files('sudalairajkumar/cryptocurrency-historical-prices-coingecko', path='./cryptoDatasets', unzip=True)

df = pd.read_csv("stockDatasets/sp500_stocks.csv", parse_dates=['Date'], index_col=['Date'])
del df["Adj Close"]
df.dropna(how='any', inplace=True)


def companyDF(company):
    tempDF = df[df['Symbol'] == company]
    tempDF = tempDF.assign(Tommorow=tempDF.Close.shift(-1))
    tempDF["Target"] = (tempDF["Tommorow"] > tempDF["Close"]).astype(int)

    # print(tempDF)
    return tempDF


AAPL = companyDF("AAPL")
MSFT = companyDF("MSFT")
AMZN = companyDF("AMZN")
GOOGL = companyDF("GOOGL")
GOOG = companyDF("GOOG")

# Training an initial machine learning model 1
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# 3.	Split the data into a train and test set

def train(company):
    train = company.iloc[:-100]
    test = company.iloc[-100:]

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    model.fit(train[predictors], train["Target"])
    RandomForestClassifier(min_samples_split=100, random_state=1)
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    print(precision_score(test["Target"], preds))

    combined = pd.concat([test["Target"], preds], axis=1)

    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:, 1]
        preds[preds >= .6] = 1
        preds[preds < .6] = 0
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined

    def backtest(data, model, predictors, start=2500, step=250):
        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i + step)].copy()
            predictions = predict(train, test, predictors, model)
            all_predictions.append(predictions)

        return pd.concat(all_predictions)

    predictions = backtest(company, model, predictors)


    print(predictions["Predictions"].value_counts())
    print(precision_score(predictions["Target"], predictions["Predictions"]))
    print(predictions["Target"].value_counts() / predictions.shape[0])
    print(predictions)


train(AAPL)
#print(AAPL)



# plot
# fig = AAPL.plot.line(y="Close", use_index=True).get_figure()
# fig.savefig('AAPL.pdf')
# plt.show()
