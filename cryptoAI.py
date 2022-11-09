from datetime import timedelta
import json
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


api.dataset_download_files('sudalairajkumar/cryptocurrency-historical-prices-coingecko', path='./cryptoDatasets', unzip=True)

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

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)




def train(crypto):
    train = crypto.iloc[:-100]
    test = crypto.iloc[-100:]

    predictors = ["price", "total_volume", "market_cap"]
    model.fit(train[predictors], train["Target"])
    RandomForestClassifier(min_samples_split=100, random_state=1)
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    #print(precision_score(test["Target"], preds))

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

    # print(predictions["Predictions"].value_counts())
    print(precision_score(predictions["Target"], predictions["Predictions"]))
    # print(predictions["Target"].value_counts() / predictions.shape[0])
    # print("Last line:")
    # print(predictions.tail(1))
    predictions = predictions.tail(1)
    print(predictions)
    return predictions

predictions1 = pd.DataFrame()



for crypto in cryptos:

    temp = train(crypto)
    predictions1 = pd.concat([predictions1, temp ])

names=['BTC', 'ETH']
predictions1['crypto'] = names

del predictions1['Target']

predictions1 = predictions1.reset_index()

predictions1['date'] = predictions1['date'].dt.strftime('%d-%m-%Y')


print("predictions:")
print(predictions1)


predictions1.loc[0].to_json("BTC.json".format(0))
predictions1.loc[1].to_json("ETH.json".format(1))


fig = BTC.plot.line(y="price", use_index=True).get_figure()
fig.savefig("BTC.pdf")
plt.show()

fig = ETH.plot.line(y="price", use_index=True).get_figure()
fig.savefig("ETH.pdf")
plt.show()