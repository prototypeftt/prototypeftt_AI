import os
import urllib
import warnings

import pandas as pd
from firebase_admin import credentials, initialize_app, storage
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

# download datasets

api = KaggleApi()
api.authenticate()

api.dataset_download_files('sudalairajkumar/cryptocurrency-historical-prices-coingecko', path='./cryptoDatasets',
                           unzip=True)

# create dataframe from csv file

btc = pd.read_csv("cryptoDatasets/bitcoin.csv", parse_dates=['date'], index_col=['date'])
btc.dropna(how='any', inplace=True)

eth = pd.read_csv("cryptoDatasets/ethereum.csv", parse_dates=['date'], index_col=['date'])
eth.dropna(how='any', inplace=True)


# create df for a crypto
def cryptoDF(df):
    df = df.assign(Tommorow=df.price.shift(-1))
    df["Target"] = (df["Tommorow"] > df["price"]).astype(int)
    df['tempAssetMovement'] = abs((df['price'] - df['Tommorow']) / df['price'] * 100)
    df = df.assign(assetMovement=df.tempAssetMovement.shift(1))
    return df


# create df for cryptos

BTC = cryptoDF(btc)
ETH = cryptoDF(eth)

# initiate lists

cryptos = [BTC, ETH]
names = ['BTC', 'ETH']
fullNames = ['Bitcoin', 'Ethereum']
closePrices = []
increasedPrices = []
decreasedPrices = []
assetMovements = []


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


# get close price

def closePrice(crypto):
    tempDF = crypto.tail(1)
    return round((tempDF.iloc[0]['price']), 2)


# get increased price

def increasedPrice(crypto):
    temp = closePrice(crypto)
    return round((temp + temp * increase(crypto)), 2)


# get decreased price

def decreasedPrice(crypto):
    temp = closePrice(crypto)
    return round((temp - temp * increase(crypto)), 2)


# get asset movement percentage between last 2 days
def assetMovement(crypto):
    tempDF = crypto.tail(1)
    return round((tempDF.iloc[0]['assetMovement']), 2)


# get the price drop or increase
def predictedPrice(row):
    if row['Predictions'] == 0:
        val = row['decreased price']
    else:
        val = row['increased price']
    return val


# populate lists

for crypto in cryptos:
    closePrices.append(closePrice(crypto))
    decreasedPrices.append(decreasedPrice(crypto))
    increasedPrices.append(increasedPrice(crypto))

# model the data

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


# create combined df for both cryptos

final = pd.DataFrame()

for crypto in cryptos:
    temp = train(crypto)
    final = pd.concat([final, temp])
    assetMovements.append(assetMovement(crypto))

final['crypto'] = names
final['close price'] = closePrices
final['increased price'] = increasedPrices
final['decreased price'] = decreasedPrices
final['assetMovement'] = assetMovements
final['predicted price'] = final.apply(predictedPrice, axis=1)

final = final.drop('decreased price', axis=1)
final = final.drop('increased price', axis=1)

del final['Target']

final = final.reset_index()
final['date'] = final['date'].dt.strftime('%d-%m-%Y')

print("predictions:")
print(final)

if not os.path.exists("predictions"):
    os.makedirs("predictions")

if not os.path.exists("graphs"):
    os.makedirs("graphs")

fig = BTC.plot.line(y="price", use_index=True).get_figure()
fig.savefig("graphs/BTC.pdf")

fig = ETH.plot.line(y="price", use_index=True).get_figure()
fig.savefig("graphs/ETH.pdf")

print("saving data to firebase database....")

url = 'https://us-central1-prototypeftt-cca12.cloudfunctions.net/api/ai/asset/add'

for i in range(0, 2):
    values = {
        'assetId': final.iloc[i]['crypto'],
        'assetCategory': 'CRYPTO',
        'assetName': fullNames[i],
        'assetClosePrice': final.iloc[i]['close price'],
        'assetPrediction': final.iloc[i]['Predictions'],
        'assetPredictedPrice': final.iloc[i]['predicted price'],
        'assetDate': final.iloc[i]['date'],
        'assetMovement': final.iloc[i]['assetMovement']

    }
    data = urllib.parse.urlencode(values)
    data = data.encode('ascii')  # data should be bytes
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as response:
        the_page = response.read()

print("saving data to firebase database....completed")

print("saving graphs to firebase storage....")

# Init firebase with your credentials
cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred, {'storageBucket': 'prototypeftt-cca12.appspot.com'})

graphs = ['graphs/BTC.pdf', 'graphs/ETH.pdf']

for graph in graphs:
    fileName = graph
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print("saving graphs to firebase storage....completed")
