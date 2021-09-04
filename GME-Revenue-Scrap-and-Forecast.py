import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

#Web scrap the data
html_data=requests.get("https://www.macrotrends.net/stocks/charts/GME/gamestop/revenue").text
soup=BeautifulSoup(html_data, 'lxml')
table=soup.find_all('tbody')[1]
gme_revenue = pd.DataFrame(columns=["Date", "Revenue"])

#Clean the data
for row in table.find_all('tr'):
    col = row.find_all("td")
    date = col[0].text
    revenue = col[1].text
    gme_revenue = gme_revenue.append({"Date":date, "Revenue":revenue}, ignore_index=True)
gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',|\$',"")
gme_revenue.dropna(inplace=True)
gme_revenue = gme_revenue[gme_revenue['Revenue'] != ""]
gme_revenue['Revenue']= gme_revenue['Revenue'].astype(int)
gme_revenue['Date']=pd.to_datetime(gme_revenue['Date'])
gme_revenue.set_index('Date', inplace=True)
gme_revenue=gme_revenue.iloc[::-1]

#analyze the data, give rough picture of the data
print(gme_revenue.describe())
gme_revenue.plot()
plt.show()

#Visualize 4 Month Rolling mean adn std
time_series=gme_revenue['Revenue']
print(type(time_series))
time_series.rolling(4).mean().plot(label='4 month rolling mean')
time_series.rolling(4).std().plot(label='4 month rolling std')
time_series.plot()
plt.legend()
plt.show()

#Visualize decompose data
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(time_series)
decomp.plot()
plt.show()
#as shown data is seasonal so we proceed with Sarima

#Augmented dickey fuller test
from statsmodels.tsa.stattools import adfuller

def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic',
              'p-value',
              '#Lags Used',
              'Number of Observations Used']

    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print(
            "strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    print()
    
adf_check(time_series)
#because the data isn't stationary, we proceed with differencing

#differencing
#first difference
gme_revenue['Revenue Diff 1']=gme_revenue['Revenue']-gme_revenue['Revenue'].shift(1)
adf_check(gme_revenue['Revenue Diff 1'].dropna())
gme_revenue['Revenue Diff 1'].plot()
plt.show()
#non-stationary

#second difference
gme_revenue['Revenue Diff 2']=gme_revenue['Revenue Diff 1']-gme_revenue['Revenue Diff 1'].shift(1)
adf_check(gme_revenue['Revenue Diff 2'].dropna())
gme_revenue['Revenue Diff 2'].plot()
plt.show()
#stationary

#seasonal difference
gme_revenue['Revenue Diff seasonal']=gme_revenue['Revenue']-gme_revenue['Revenue'].shift(4)
adf_check(gme_revenue['Revenue Diff seasonal'].dropna())
gme_revenue['Revenue Diff seasonal'].plot()
plt.show()
#stationary

#autocorrelation and Partial Autocorrelaction plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig_first = plot_acf(gme_revenue["Revenue Diff seasonal"].dropna())
fig_second = plot_pacf(gme_revenue["Revenue Diff seasonal"].dropna())
plt.show()
#because using seasonal differencing is easier to determine arima order with acf and pacf plot, we proceed with using seasonal differencing
#arima(0,4,3) used as shown in acf and pacf plot

#making the model
model=sm.tsa.statespace.SARIMAX(gme_revenue['Revenue'],order=(0,4,3),seasonal_order = (1,1,3,4))
hasil=model.fit()
print(hasil.summary())
hasil.resid.plot(kind='kde')

#model test
gme_revenue['model test']=hasil.predict(start=55, end=66, dynamic=True)
gme_revenue[['Revenue','model test']].plot()
plt.show()

#forecast
future_prediction=hasil.predict(start=66,end=73)
print(future_prediction)
future_prediction.plot()
gme_revenue['Revenue'].plot()
plt.show()
