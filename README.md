# Gold Regression
Multivariate regression model on gold price

## Overview
* Linear regression model, and multivariate regression model built on Python for gold price valuation
* Data collected from FRED and EIA by calling API and parsing JSON objects
* Data processed using pandas, modelled using scikit-learn and visualised with seaborn
* 15 years worth of gold price was modelled with a R-squared value of 0.87 using fundamentally-sound macroeconomic data

## Data Collection
* Historical gold prices were downloaded from Yahoo Finance using `yfinance` library
* Data for Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity was collected from **St. Louis Fed**, retrieving JSON object by API call
* Data for ending crude oil stocks in Cushing Oklahoma (WTI storage) was collected from **U.S. EIA**, also by retrieving JSON object from API call

## Data Cleaning
* Non numerical data was observed from FRED's data, which was converted to `np.nan`
* Attempted to perform data imputation by K nearest neighbor method with `KNNImputer` in `sklearn` library. Imputed data creates unncessary variance, which might be due to limited number of attributes (features)
* Null data was eventually dropped, as they only contributes to <5% of total data
* Following shows the data I was working with, real yield (%) with a frequency of daily, and crude oil ending stock (thousand barrals) with a frequency of weekly   
![real_yield](https://user-images.githubusercontent.com/106392189/172883099-6f9328b0-8fe4-4458-adbc-cf6fe86d5d46.png)
![cushingOK](https://user-images.githubusercontent.com/106392189/172881913-5df5d913-a498-453e-8f28-1b1105f1bd51.png)
* And the gold prices in USD:  
![gold](https://user-images.githubusercontent.com/106392189/172880725-c9e9519d-73a6-4b22-bfa6-c6a74bd99ccb.png)
  
## Reasons for the choice of predictor variables
* From a **fundamental** perspective, gold has long been considered as an inflation hedge and a hedge against fiat currencies. Whereas inflation adjusted treasury yield (real yield) can be considered as returns on investing in USD. Therefore, it is expected that these 2 should be inversely-correlated (given gold prices are measured in USD terms)
* Since Nixon ended the USD-gold peg in 1971, USD has entered into a new era of petrodollar system where USD is "pseudo-pegged" with crude oil. While US started as an oil-importing country, it became a net-exporting country after the 2006 shale revolution (albeit hugely affected by green polocies in recent years). My macro thesis is therefore: the more stock US holds in crude, the more it can export thus raising demand for USD among oil-importing countries; as USD strength is inversely-related to gold/$, crude stock should also be inversely-related against gold price
* From a **technical** perspective, an inversed version of the "cup and handle" pattern in gold was observed in both real yield and crude stock graphs as in above section, starting from mid-2014 to 2022

## Modelling
* Based on above analysis, range of data was determined to be starting from 2006 till now
* [**Linear regression model**](https://github.com/urinethrower/gold_regression/blob/main/linear%20gold.py) (X = real yield; y = gold price; frequency = daily) was fitted first, measured **R-sqr = 0.86**, **RMSE = $215.01**
* [**Multivariate regression model**](https://github.com/urinethrower/gold_regression/blob/main/multivar%20reg%20gold.py) (X = real yield, Cushing crude oil storage; y = gold price; frequency = weekly) was fitted next hoping to improve on modelling quality. Measured **R-sqr = 0.87**, **RMSE = $220.43**
* Comparison between actual and predicted gold prices for both models are plotted as follows:  
![linear model](https://user-images.githubusercontent.com/106392189/172914317-112b4cbe-3886-4eb2-aef5-bc9517590a02.png)
![multi reg model](https://user-images.githubusercontent.com/106392189/172914325-e4a0a4a1-10a9-4e12-85bd-3b1b5bca2b58.png)

## Resources Used
* Python library packages: yfinance, sklearn, numpy, pandas, json, seaborn & matplotlib
* API provided by Federal Reserve Economic Data from St. Louis Fed, U.S. Energy Information Administration
* Historical price data downloaded from Yahoo Finance
