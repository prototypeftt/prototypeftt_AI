import os
import warnings

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

api = KaggleApi()
api.authenticate()

api.dataset_download_files('andrewmvd/sp-500-stocks', path='stockDatasets', unzip=True)

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
    # print(precision_score(test["Target"], preds))

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

    predictions = predictions.tail(1)

    return predictions


companies = [AAPL, MSFT, AMZN, GOOGL, GOOG]
names = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG']

predictions1 = pd.DataFrame()

for company in companies:
    temp = train(company)
    predictions1 = pd.concat([predictions1, temp])

predictions1['stock'] = names

predictions1 = predictions1.reset_index()

predictions1['Date'] = predictions1['Date'].dt.strftime('%d-%m-%Y')
del predictions1['Target']
print("predictions:")
print(predictions1)

if not os.path.exists("predictions"):
    os.makedirs("predictions")

if not os.path.exists("graphs"):
    os.makedirs("graphs")

predictions1.loc[0].to_json("predictions/AAPL.json".format(0))
predictions1.loc[1].to_json("predictions/MSFT.json".format(1))
predictions1.loc[2].to_json("predictions/AMZN.json".format(2))
predictions1.loc[3].to_json("predictions/GOOGL.json".format(3))
predictions1.loc[4].to_json("predictions/GOOG.json".format(4))

fig = AAPL.plot.line(y="Close", use_index=True).get_figure()
fig.savefig("graphs/AAPL.pdf")

fig = MSFT.plot.line(y="Close", use_index=True).get_figure()
fig.savefig("graphs/MSFT.pdf")

fig = AMZN.plot.line(y="Close", use_index=True).get_figure()
fig.savefig("graphs/AMZN.pdf")

fig = GOOGL.plot.line(y="Close", use_index=True).get_figure()
fig.savefig("graphs/GOOGL.pdf")

fig = GOOG.plot.line(y="Close", use_index=True).get_figure()
fig.savefig("graphs/GOOG.pdf")
