import os
import warnings

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('mode.chained_assignment', None)

api = KaggleApi()
api.authenticate()

api.dataset_download_files('sudalairajkumar/cryptocurrency-historical-prices-coingecko', path='./cryptoDatasets',
                           unzip=True)

btc = pd.read_csv("cryptoDatasets/bitcoin.csv", parse_dates=['date'], index_col=['date'])
btc.dropna(how='any', inplace=True)

eth = pd.read_csv("cryptoDatasets/ethereum.csv", parse_dates=['date'], index_col=['date'])
eth.dropna(how='any', inplace=True)


def cryptoDF(df):
    df = df.assign(Tommorow=df.price.shift(-1))
    df["Target"] = (df["Tommorow"] > df["price"]).astype(int)

    # print(tempDF)
    return df


BTC = cryptoDF(btc)
ETH = cryptoDF(eth)

cryptos = [BTC, ETH]
names = ['BTC', 'ETH']

closePrices = []
incresedPrices = []
decresedPrices = []


# last 30 days mean of increase or drop

def increase(crypto):
    tempDF = crypto.tail(30)
    tempDF['Difference'] = tempDF['price'] - tempDF['Tommorow']
    tempDF['Percentage'] = tempDF['Difference'] / tempDF['price']
    tempDF = tempDF[tempDF['Percentage'] >= 0]

    return tempDF["Percentage"].mean()


def decrease(crypto):
    tempDF = crypto.tail(30)
    tempDF['Difference'] = tempDF['price'] - tempDF['Tommorow']
    tempDF['Percentage'] = tempDF['Difference'] / tempDF['price'] * (-1)
    tempDF = tempDF[tempDF['Percentage'] >= 0]

    return tempDF["Percentage"].mean()


def closePrice(crypto):
    tempDF = crypto.tail(1)
    return round((tempDF.iloc[0]['price']), 2)


def increasedPrice(crypto):
    temp = closePrice(crypto)
    return round((temp + temp * increase(crypto)), 2)


def decreasedPrice(crypto):
    temp = closePrice(crypto)
    return round((temp - temp * increase(crypto)), 2)


for crypto in cryptos:
    closePrices.append(closePrice(crypto))
    decresedPrices.append(decreasedPrice(crypto))
    incresedPrices.append(increasedPrice(crypto))

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


def train(crypto):
    train = crypto.iloc[:-100]
    test = crypto.iloc[-100:]

    predictors = ["price", "total_volume", "market_cap"]
    model.fit(train[predictors], train["Target"])
    RandomForestClassifier(min_samples_split=100, random_state=1)
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
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

    predictions = backtest(crypto, model, predictors)

    predictions = predictions.tail(1)

    return predictions


predictions1 = pd.DataFrame()

for crypto in cryptos:
    temp = train(crypto)
    predictions1 = pd.concat([predictions1, temp])

predictions1['crypto'] = names

predictions1['close price'] = closePrices

predictions1['increased price'] = incresedPrices

predictions1['decreased price'] = decresedPrices


def predictedPrice(row):
    if row['Predictions'] == 0:
        val = row['decreased price']
    else:
        val = row['increased price']
    return val


predictions1['predicted price'] = predictions1.apply(predictedPrice, axis=1)

predictions1 = predictions1.drop('decreased price', axis=1)
predictions1 = predictions1.drop('increased price', axis=1)

del predictions1['Target']

predictions1 = predictions1.reset_index()

predictions1['date'] = predictions1['date'].dt.strftime('%d-%m-%Y')

print("predictions:")
print(predictions1)

if not os.path.exists("predictions"):
    os.makedirs("predictions")

if not os.path.exists("graphs"):
    os.makedirs("graphs")

predictions1.loc[0].to_json("predictions/BTC.json".format(0))
predictions1.loc[1].to_json("predictions/ETH.json".format(1))

fig = BTC.plot.line(y="price", use_index=True).get_figure()
fig.savefig("graphs/BTC.pdf")

fig = ETH.plot.line(y="price", use_index=True).get_figure()
fig.savefig("graphs/ETH.pdf")
