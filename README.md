1.Web scrapping GameStop Revenue in https://www.macrotrends.net/stocks/charts/GME/gamestop/revenue

2.visualize 4 month rolling mean and std (if data monthly then 12 month rolling, my data is quarterly)

3.decompose data to indentify if the data seasonal or not

4.clean the data and proceed to check stationary data with augmented dickey fuller test

5.use differencing to find stationary data

6.plot acf and pacf to determine arima order (sarima if the data seasonal in step 2)

7.make the model, test the model with the data set

8.plot model residual to see if error spread in 0 or not

9.forecast future revenue
