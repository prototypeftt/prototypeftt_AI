import os
import urllib
import warnings

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('mode.chained_assignment', None)

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

companies = [AAPL, MSFT, AMZN, GOOGL, GOOG]
names = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG']

closePrices = []
incresedPrices = []
decresedPrices = []


# last 30 days mean of increase or drop

def increase(company):
    tempDF = company.tail(30)
    tempDF['Difference'] = tempDF['Close'] - tempDF['Tommorow']
    tempDF['Percentage'] = tempDF['Difference'] / tempDF['Close']
    tempDF = tempDF[tempDF['Percentage'] >= 0]

    return tempDF["Percentage"].mean()


def decrease(company):
    tempDF = company.tail(30)
    tempDF['Difference'] = tempDF['Close'] - tempDF['Tommorow']
    tempDF['Percentage'] = tempDF['Difference'] / tempDF['Close'] * (-1)
    tempDF = tempDF[tempDF['Percentage'] >= 0]

    return tempDF["Percentage"].mean()


def closePrice(company):
    tempDF = company.tail(1)
    return round((tempDF.iloc[0]['Close']), 2)


def increasedPrice(company):
    temp = closePrice(company)
    return round((temp + temp * increase(company)), 2)


def decreasedPrice(company):
    temp = closePrice(company)
    return round((temp - temp * increase(company)), 2)


for company in companies:
    closePrices.append(closePrice(company))
    decresedPrices.append(decreasedPrice(company))
    incresedPrices.append(increasedPrice(company))

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


predictions1 = pd.DataFrame()

for company in companies:
    temp = train(company)
    predictions1 = pd.concat([predictions1, temp])

predictions1['stock'] = names
# added
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

predictions1 = predictions1.reset_index()

predictions1['Date'] = predictions1['Date'].dt.strftime('%d-%m-%Y')
del predictions1['Target']

if not os.path.exists("predictions"):
    os.makedirs("predictions")

if not os.path.exists("graphs"):
    os.makedirs("graphs")

properNames = ["Apple", "Microsoft", "Amazon", "Alphabet1", "Alphabet2"]

final = pd.DataFrame()

final['assetId'] = predictions1['stock']
final['assetCategory'] = "STOCK"
final['assetName'] = properNames
final['assetClosePrice'] = predictions1['close price']
final['assetPrediction'] = predictions1['Predictions']
final['assetPredictedPrice'] = predictions1['predicted price']
final['assetDate'] = predictions1['Date']
final.set_index('assetId')

print("final:")
print(final)

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

print("calls")

url = 'https://us-central1-prototypeftt-cca12.cloudfunctions.net/api/ai/asset/add'

for i in range(0, 5):
    values = {
        'assetId': final.iloc[i]['assetId'],
        'assetCategory': final.iloc[i]['assetCategory'],
        'assetName': final.iloc[i]['assetName'],
        'assetClosePrice': final.iloc[i]['assetClosePrice'],
        'assetPrediction': final.iloc[i]['assetPrediction'],
        'assetPredictedPrice': final.iloc[i]['assetPredictedPrice'],
        'assetDate': final.iloc[i]['assetDate']

    }
    data = urllib.parse.urlencode(values)
    data = data.encode('ascii')  # data should be bytes
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as response:
        the_page = response.read()

print("calls end")

from firebase_admin import credentials, initialize_app, storage

# Init firebase with your credentials
cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred, {'storageBucket': 'prototypeftt-cca12.appspot.com'})

graphs = ['graphs/AAPL.pdf', 'graphs/AMZN.pdf', 'graphs/GOOG.pdf', 'graphs/GOOGL.pdf', 'graphs/MSFT.pdf', ]

# Put your local file path

for graph in graphs:
    fileName = graph
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

    # Opt : if you want to make public access from the URL
