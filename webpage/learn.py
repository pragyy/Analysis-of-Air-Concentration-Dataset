# Importing required packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hurst import compute_Hc, random_walk
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import warnings
import copy
import matplotlib as mpl
import pymannkendall as mk
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#-------------------------------------------------------------------------------------------#

# Function to cache and load the dataset
@st.cache
def load_data():
    df = pd.read_csv('abc.csv', parse_dates=['date'], index_col = 'date')
    return df
# Loading the dataset
df = load_data()
df1 = pd.read_csv('abc.csv', parse_dates = ['date'])
aqi_df = pd.read_csv('aqi.csv', parse_dates = ['date'], index_col = 'date')
#-------------------------------------------------------------------------------------------#

# Main function / the navigation bar
def main():
    
    st.sidebar.title("INDEX")
    page = st.sidebar.selectbox("CONTENTS",["Homepage", "Data Preprocessing","PCA", "Data Visualization", 
    										"Trend Analysis", "Data Predictibility","AQI Calculation", 
    										"Stationarity Check"])

    if page == "Homepage":
    	st.title('ANALYSIS OF AIR QUALITY DATA')
    	st.header("INTRODUCTION")
    	homepage()

    elif page == "Data Preprocessing":
    	st.title("PREPROCESSING THE DATASET")
    	preprocessing()        

    elif page == "Data Visualization":
        st.title("DATA VISUALIZATION")
        visualization()

    elif page == "Trend Analysis":
    	st.title("DECOMPOSITION OF DATASET")
    	trend()

    elif page == "Data Predictibility":
    	st.title("DATA PREDICTIBILITY")
    	data_predictability()

    elif page == "AQI Calculation":
    	st.title('AIR QUALITY INDEX')
    	aqi()

    elif page == "Stationarity Check":
    	st.title("CHECKING STATIONARITY OF POLLUTANT")
    	stationarity_page()


#-------------------------------------------------------------------------------------------#
#---------------------------------------HOME PAGE-------------------------------------------#

# Defigning Homepage
def homepage():
	st.write("It is fairly apparent about the rate at which the quality of air is declining in the current environment extensively in the urban areas. One such city, where the poor quality of air can clearly be identified is the capital city of India, Delhi. The Air Quality Index (AQI) is a powerful tool on the basis of which the characteristics of air can be determined in a certain area. The AQI for the city of Delhi is computed by monitoring the four main pollutants namely nitrogen dioxide (NO2), sulphur dioxide (SO2), suspended particulate matter (SPM),and residual suspended particulate matter (RSPM) by calculating the air quality indices for these pollutants. With every country following a different scale for evaluation, the values provided by the Central Pollution Control Board of India are used to assess the condition of air of the region under consideration. The Seasonal and Daily calculation of AQI divulged the quality of air in the study region which could further be classified into various sections stretching across good, satisfactory, moderately polluted, poor,very poor and severe based on the AQI that was estimated.")
	st.subheader("Project Outline")
	st.write("In this analysis project, we aim to develop a forecasting model to predict the air quality index. The projects involves various techniques for data wrangling. Various statistical tests like Mann-Kendall and ADF are also used to determine the stationarity and trend in the data. Finally, an ARIMA Foreasting model is developed to serve the purpose of this project.")
	st.subheader("About Dataset")
	st.write("The dataset used is a time-series data. The dataset under consideration consists of four main pollutants viz, NO2, SO2, SPM and RSPM using which the Air Quality Index is calculated. It contains daily concentrations of the aforementioned pollutants from 2008-2010 at the Hazrat Nizamuddin Railway Station, Delhi. ")
	st.write("The dataset is depicted as follows: ")
	st.write(df)
	st.subheader("About Project")
	st.write("This project is carried out under the organization Bhabha Atomic Research Center(BARC).")

#-------------------------------------------------------------------------------------------#
#----------------------------------PREPROCESSING PAGE---------------------------------------#

# Defining Preprocessing Page
def preprocessing():
	st.header("RAW DATA")
	st.write("This is how the dataset looked like before processing")
	unclean = pd.read_excel('uncleaned.xlsx')
	st.write(unclean)
	st.write("The dataset has a lot of empty cells. This occurs due to variety of reasons during data collection viz, the values may be too minimal to be detected or the device might have some problems while recording the data.")

	st.header("DATA CLEANING")
	st.write("Before handling the missing cells, we have merged the date, month and year column into a single column \"date\" and converted into a datetime format so that it becomes easy to handle the timeseries dataset.")
	st.write("The real world data have a lot of missing cells. One way to handle this problem is to get rid of the observations that have missing data. However, there will be risk of losing data points with valuable information. A better strategy is to impute the missing values. In other words, to infer those missing values from the existing part of the data.")
	st.write("Here we have used mean imputation technique to get rid of the empty cells. This works by calculating the mean/median of the non-missing values in a column and then replacing the missing values within each column separately and independently from the others. It can only be used with numeric data.")
	st.write("After all the cleaning tasks are performed, the dataset looks as below: ")
	st.write(df)

	st.header("VERIFYING DATA CLEANLINESS")
	st.write("Before proceeding to use the data set we need to check whether the data is fit enough for the analysis and is properly cleaned.")
	verify_clean()

# Defining the function to verify the columns
def verify_clean():

	st.subheader("Checking for date ")
	st.write("All dates are stored in YYYY-MM-DD format.")
	st.write("Values are recorded on daily basis.")

	st.subheader("Checking for pollutants ")
	option = st.selectbox('For which column do you want to check?', ('so2', 'no2', 'spm', 'rspm'))

	x = pd.Series([option])

	# Checking for null values
	null_check = x.all()
	if null_check:
		'''Sum of ***is-null*** function is zero here because all the cells are entered with mean/median imputated data. Thus all the cells are correctly filled.'''
	else:
		st.write("Improper handling of missing values since empty cells still exists.")

	#Checking datatype and sign of all concentration
	flag = 0
	for i in x:
		if (type(i)==float):
			continue
		else:
			flag = 1
			st.write("Values are of float data type and non negative.")

	if(flag==0):
		st.write("All values stored are of float data type and non negative.")

#-------------------------------------------------------------------------------------------#
#-----------------------------------VISUALIZATION PAGE--------------------------------------#

# Defining the visualization page
def visualization():
	st.header("LINE PLOTTING")
	st.write("Daily concentrations of each pollutant are plotted to get an insight about the dataset.")
	line_plot()

	st.header("HISTOGRAM PLOTTING")
	st.write("The following histogram plots depict the values of concentration which has high frequency of occurence.")
	histogram()

	st.header("SCATTER PLOTTING")
	st.write("Pollutants are plotted against each other.")
	scatter_plot()

# function for line plotting
def line_plot():
	fig = go.Figure()
	fig.add_trace(go.Scatter( x=df1.date,y=df1['no2'],name="NO2",line_color='deepskyblue',opacity=0.8))
	fig.add_trace(go.Scatter( x=df1.date,y=df1['so2'],name="SO2",line_color='dimgray',opacity=0.8))
	fig.add_trace(go.Scatter(x=df1.date,y=df1['rspm'],name="RSPM",line_color='green',opacity=0.8))
	fig.add_trace(go.Scatter(x=df1.date,y=df1['spm'],name="SPM",line_color='navy',opacity=0.8))
	fig.update_layout(xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

#function for histogram plotting
def histogram():
	fig = make_subplots(rows=2, cols=2)
	no2 = go.Histogram(x=df1['no2'], nbinsx=4)
	so2 = go.Histogram(x=df1['so2'], nbinsx=4)
	spm = go.Histogram(x=df1['spm'], nbinsx=4)
	rspm = go.Histogram(x=df1['rspm'], nbinsx=4)
	fig.append_trace(no2, 1, 1)
	fig.append_trace(so2, 1, 2)
	fig.append_trace(spm, 2, 1)
	fig.append_trace(rspm, 2, 2)
	st.plotly_chart(fig)

# function for scatter plotting
def scatter_plot():
	fig = px.scatter_matrix(df1, dimensions=["no2", "so2", "rspm", "spm"])
	st.plotly_chart(fig)	

#-------------------------------------------------------------------------------------------#
#-------------------------------DATA PREDICTIBILITY PAGE------------------------------------#

# Defining the data predictibility page
def data_predictability():
	pred = st.sidebar.radio('Parameters to judge the predictibility',('Hurst Exponent','Fractal Dimension','Power Law'))

	if pred == 'Hurst Exponent':
		hurst_exponent()
	elif pred == 'Fractal Dimension':
		fractal()
	elif pred == 'Power Law':
		power_law()

# Function for hurst exponent
def hurst_exponent():
	st.header("HURST EXPONENT")
	st.write('The Hurst Exponent includes the following concepts: ')
	'''**Brownian Time Series:** In a Brownian time series (also known as a random walk or a drunkard’s walk) there is no correlation between the observations and a future observation; being higher or lower than the current observation are equally likely. Series of this kind are hard to predict. Hurst exponent close to **0.5** is indicative of a Brownian time series.''' 
	'''**Anti-persistent Time Series:** In an anti-persistent time series (also known as a mean-reverting series) an increase will most likely be followed by a decrease or vice-versa (i.e., values will tend to revert to a mean). This means that future values have a tendency to return to a long-term mean. Hurst exponent value between 0 and 0.5 is indicative of anti-persistent behavior and the **closer the value is to 0**, the stronger is the tendency for the time series to revert to its long-term means value.'''  
	'''**Persistent Time Series:** In a persistent time series an increase in values will most likely be followed by an increase in the short term and a decrease in values will most likely be followed by another decrease in the short term. Hurst exponent value **between 0.5 and 1.0** indicates persistent behavior; the larger the H value the stronger the trend."'''
	
	st.subheader("Hurst exponent calculation")
	pollutant = st.selectbox('Which pollutant to explore?',('NO2', 'SO2', 'RSPM', 'SPM'))
	'''The *slope* of the following plot is used to calculate the Hurst Exponent, where the *x-axis* represent the time interval and *y-axis* represent the R/S (Rescaled Range) Ratio. The resaled range of time series is calculated from dividing the range of its mean adjusted cumulative deviate series by the standard deviation of the time series itself.'''
	if pollutant == 'NO2':
		hurst(df['no2'])
		'''**Anti Persistent Time Series**'''
	elif pollutant == 'SO2':
		hurst(df['so2'])
		'''**Anti Persistent Time Series**'''
	elif pollutant == 'RSPM':
		hurst(df['rspm'])
		'''**Persistent Time Series**'''
	elif pollutant == 'SPM':
		hurst(df['spm'])
		'''**Persistent Time Series**'''

# Fucntion to calculate hurst for each pollutant
def hurst(df_pollutant):
	H, c, data = compute_Hc(df_pollutant, kind='price')
	f, ax = plt.subplots()
	ax.plot(data[0], c*data[0]**H, color="deepskyblue")
	ax.scatter(data[0], data[1], color="purple")
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('Time interval')
	ax.set_ylabel('R/S ratio')
	ax.grid(True)
	st.pyplot()
	st.write("H={:.4f}, c={:.4f}".format(H,c)) 

# Function to calculate Fractal Dimension
def fractal():
	st.header('FRACTAL DIMENSION')
	
	st.subheader('Fractal Dimension for Nitrogen dioxide')
	H_no2, c, data = compute_Hc(df['no2'], kind='price')
	D_no2 = 2 - H_no2
	st.write(D_no2)

	st.subheader('Fractal Dimension for Sulphur dioxide')
	H_so2, c, data = compute_Hc(df['so2'], kind='price')
	D_so2 = 2 - H_so2
	st.write(D_so2)

	st.subheader('Fractal Dimension for Suspended Particulate Matter')
	H_spm, c, data = compute_Hc(df['spm'], kind='price')
	D_spm = 2 - H_spm
	st.write(D_spm)

	st.subheader('Fractal Dimension for Residual Suspended Particulate Matter')
	H_rspm, c, data = compute_Hc(df['rspm'], kind='price')
	D_rspm = 2 - H_rspm
	st.write(D_rspm)

# Function to calculate power law
def power_law():
	st.write('b = 2*H + 1 = 5 - 2*D')
	st.write('where,  D = Fractal Dimension and H = Hausdorff Measure')

	'''here **b** represents ***beta***  
	if b = 0  
	white noise, uncorrelated and power spectrum independant of frequency.  
	if b = 1  
	flicker of 1/f noise systems, moderately correlated.  
	if b = 2  
	brownian noise like systems, strongly correlated.'''

	st.subheader('Beta for Nitrogen dioxide')
	H_no2, c, data = compute_Hc(df['no2'], kind='price')
	D_no2 = 2 - H_no2
	b_no2 = 5 - (2 * D_no2)
	st.write(b_no2)

	st.subheader('Beta for Sulphur dioxide')
	H_so2, c, data = compute_Hc(df['so2'], kind='price')
	D_so2 = 2 - H_so2
	b_so2 = 5 - (2 * D_so2)
	st.write(b_so2)

	st.subheader('Beta for Suspended Particulate Matter')
	H_spm, c, data = compute_Hc(df['spm'], kind='price')
	D_spm = 2 - H_spm
	b_spm = 5 - (2 * D_spm)
	st.write(b_spm)

	st.subheader('Beta for Residual suspended Particulate Matter')
	H_rspm, c, data = compute_Hc(df['rspm'], kind='price')
	D_rspm = 2 - H_rspm
	b_rspm = 5 - (2 * D_rspm)
	st.write(b_rspm)

#-------------------------------------------------------------------------------------------#
#-----------------------------------DECOMPOSITION PAGE--------------------------------------#

# Function for additive decomposition from scratch
def additive_decomposition(df_pollutant):

	st.header('DEFINING TREND OUR DATASET')

	# Theoritecal trend
	st.sidebar.subheader("Theoretical Trend")
	order = st.sidebar.slider('Select the order of polynomial', min_value = 0, max_value = 15, step = 1) #degree of the polynomial defining our trend
	# coef variable storing all the coefficient values of the polynomial
	coef = np.polyfit(np.arange(len(df_pollutant)), df_pollutant.values.ravel(), order)
	poly_mdl = np.poly1d(coef)
	trend = pd.Series(data = poly_mdl(np.arange(len(df_pollutant))), index = df.index)
	st.subheader('Plotting actual dataset with the theoretical trend')
	'''*NOTE: The theoretical trend is directly dependent on the order of the polynomial *'''
	df_pollutant.plot()
	trend.plot()
	st.pyplot()

	# Dtrended dataset
	st.header('DETRENDING THE DATASET')
	detrended = df_pollutant - trend
	plt.plot(detrended)
	st.pyplot()

	# Plotting deterended dataset
	st.subheader('Grouping detrended data and plotting it yearly')
	yearly_data = detrended.groupby(by = detrended.index.year)
	for year in yearly_data.groups:
		plt.plot(yearly_data.get_group(year).index.month, yearly_data.get_group(year).values, label=year)
		axes = plt.gca()
		axes.set_xlim([0,13])
		plt.grid()
		plt.legend()
		plt.show()
		st.pyplot()

	# Monthly Plotting deterended dataset
	st.subheader('Monthly mean plotting of the detrended data')
	seasonal = detrended.groupby(by = detrended.index.month).mean()
	plt.plot(seasonal, label='Mean Seasonality in a year')
	plt.grid()
	plt.legend()
	st.pyplot()

	# Monthly Plotting for each year of deterended dataset
	st.subheader('Plotting monthly mean dataset for all years') 
	yearly_data = detrended.groupby(by = detrended.index.year)
	for year in yearly_data.groups:
		plt.plot(yearly_data.get_group(year).index.month, yearly_data.get_group(year).values, label=year)
	plt.plot(seasonal, color="black", label='Mean Seasonality in a year')
	plt.grid()
	plt.legend()
	st.pyplot()

	# Removing seasonality
	st.header('REMOVING SEASONALITY')
	st.subheader('plotting seasonality of the dataset')
	#rcParams['figure.figsize'] = 18,5
	seasonal_component = copy.deepcopy(df_pollutant)
	for date in seasonal.index:
		seasonal_component.loc[seasonal_component.index.month == date] = seasonal.loc[date]
	plt.plot(seasonal_component)
	plt.grid()
	st.pyplot()

	# Plotting deseasonal dataset
	st.subheader("Plotting deseasonal data")
	deseasonal = df_pollutant - seasonal_component
	plt.plot(deseasonal)
	st.pyplot()

	# Extracting trend from deseasonal data
	st.subheader('Extracting trend from deseasonal data')
	st.sidebar.subheader('Deseasonal data')
	order = st.sidebar.slider('Select the order of polynomial ', min_value = 0, max_value = 15, step = 1)
	coef = np.polyfit(np.arange(len(deseasonal)),deseasonal.values.ravel(),order)
	poly_mdl = np.poly1d(coef)
	trend_component = pd.DataFrame(data = poly_mdl(np.arange(len(df_pollutant))),index = df.index, columns=['pollutant'])
	trend_component.plot()
	plt.title('Trend component extracted from deseasonal data')
	plt.xlabel('Year')
	plt.ylabel('Values')
	plt.grid()
	st.pyplot()

	st.subheader('Comparing theoritical trend with the actual trend of deseasonal data')  
	plt.plot(trend, color='blue', label='Theoretical Model')
	plt.plot(trend_component, color='orange', label='Actual Trend in the Deseasonal Data')
	plt.xlabel('Year')
	plt.ylabel('Values')
	plt.grid()
	plt.legend()
	st.pyplot()

# Additive decomposition page
def trend_analysis():
	pol = st.sidebar.selectbox('Select the pollutant whose trend you want to analyse',('NO2','SO2','SPM','RSPM'))

	if pol == 'NO2':
		additive_decomposition(df['no2'])

	if pol == 'SO2':
		additive_decomposition(df['so2'])

	if pol == 'SPM':
		additive_decomposition(df['spm'])

	if pol == 'RSPM':
		additive_decomposition(df['rspm'])

# Function for additive and multiplicative decoposition
def decomposition_fun(df_pollutant,frequency):
	# Multiplicative Decomposition 
	result_mul_so2 = seasonal_decompose(df_pollutant, model='multiplicative', freq = frequency)

	# Additive Decomposition
	result_add_so2 = seasonal_decompose(df_pollutant, model='additive', freq = frequency)

	#plt.rcParams.update({'figure.figsize': (10,10)})
	st.subheader('Multiplicative Decomposition')
	result_mul_so2.plot()
	st.pyplot()
	st.subheader('Additive Decomposition')
	result_add_so2.plot()
	st.pyplot()

# Additive nd multiplicative decomposition page
def decomposition():
	st.header('Additive and Multiplicative Decomposition')
	st.sidebar.subheader('Additive and Multiplicative Decomposition')
	poll = st.sidebar.selectbox('Select the pollutant',('no2','so2','spm','rspm'))
	q = st.sidebar.radio('Select the frequency',('Weekly','Monthly','Yearly'))
	
	if poll == 'no2':
		st.subheader('Decomposition for Nitrogen Dioxide')
		if q == 'Weekly':
			decomposition_fun(df['no2'],7)
		elif q == 'Monthly':
			decomposition_fun(df['no2'],31)
		elif q == 'Yearly':
			decomposition_fun(df['no2'],365)

	elif poll == 'so2':
		st.subheader('Decomposition for Sulphur Dioxide')
		if q == 'Weekly':
			decomposition_fun(df['so2'],7)
		elif q == 'Monthly':
			decomposition_fun(df['so2'],31)
		elif q == 'Yearly':
			decomposition_fun(df['so2'],365)

	elif poll == 'spm':
		st.subheader('Decomposition for Suspended Particulate Matter')
		if q == 'Weekly':
			decomposition_fun(df['spm'],7)
		elif q == 'Monthly':
			decomposition_fun(df['spm'],31)
		elif q == 'Yearly':
			decomposition_fun(df['spm'],365)

	elif poll == 'rspm':
		st.subheader('Decomposition for Residual Suspended Particulate Matter')
		if q == 'Weekly':
			decomposition_fun(df['rspm'],7)
		elif q == 'Monthly':
			decomposition_fun(df['rspm'],31)
		elif q == 'Yearly':
			decomposition_fun(df['rspm'],365)

# Function for mann kendall test
def mann_kendall():
	st.header('MANN KENDALL TREND ANALYSIS')

	'''The Mann-Kendall Trend Test (sometimes called the MK test) is used to analyze time series data for consistently 
	increasing or decreasing trends (monotonic trends). It is a non-parametric test, which means it works for all 
	distributions (i.e. data doesn't have to meet the assumption of normality), but data should have no serial correlation.
	If the data has a serial correlation, it could affect in significant level (p-value). It could lead to misinterpretation.
	To overcome this problem, researchers proposed several modified Mann-Kendall tests (Hamed and Rao Modified MK Test, 
	Yue and Wang Modified MK Test, Modified MK test using Pre-Whitening method, etc.). Seasonal Mann-Kendall test also 
	developed to remove the effect of seasonality. '''
	'''Mann-Kendall Test is a powerful trend test, so several others modified Mann-Kendall tests like Multivariate MK Test, 
	Regional MK Test, Correlated MK test, Partial MK Test, etc. were developed for the spacial condition. '''

	''' Mann-Kendall tests return a named tuple which contains:  
	***Trend: *** Tells the trend (increasing, decreasing and no trend)  
	***h: *** True (if trend is present) or False (if trend is absent)  
	***p: *** p-value of the significance test  
	***z: *** normalized test statistics  
	***Tau: *** Kendall Tau
	***s: *** Mann-Kendall's Score  
	***var_s: *** Variance S  
	***slope: *** Sen's slope
	'''

	poll = st.sidebar.selectbox('Choose the pollutant',('no2','so2','spm','rspm'))

	if poll == 'so2':
		st.subheader('Trend Analysis for Sulphur Dioxide')
		result = mk.original_test(df.so2)
		st.write(str(result))

	elif poll == 'no2':
		st.subheader('Trend Analysis for Nitrogen Dioxide')
		result = mk.original_test(df.no2)
		st.write(str(result))

	elif poll == 'spm':
		st.subheader('Trend Analysis for Suspended Particulate Matter')
		result = mk.original_test(df.spm)
		st.write(str(result))

	elif poll == 'rspm':
		st.subheader('Trend Analysis for Residual Suspended Particulate Matter')
		result = mk.original_test(df.rspm)
		st.write(str(result))

# Defining trend analysis Main Page
def trend():
	pg = st.sidebar.selectbox('Choose the trend analysis',('Decomposition', 'Stepwise Additive Decomposition', 'Mann-Kendall Analysis'))

	if pg == 'Decomposition':
		decomposition()

	elif pg == 'Stepwise Additive Decomposition':
		trend_analysis()

	elif pg == 'Mann-Kendall Analysis':
		mann_kendall() 

#-------------------------------------------------------------------------------------------#
#----------------------------------------AQI PAGE-------------------------------------------#

def aqi():
	date = st.date_input("Enter the day of which you want to see calculated AQI (2008-2010): ")
	asked_aqi = aqi_df.loc[date]['aqi']
	st.write('The required AQI value is: ', asked_aqi)

#-------------------------------------------------------------------------------------------#
#------------------------------------STATIONARITY PAGE--------------------------------------#

def stationarity_check(df_pollutant):
	st.header("METHOD 1: PLOTTING")
	df_pollutant.plot()
	st.pyplot()

	st.header("METHOD 2: STATISTICAL SUMMARY")
	X = df_pollutant
	#X = log(X)
	split = round(len(X) / 6)
	X1, X2, X3, X4, X5, X6 = X[0:split], X[split+1:2*split], X[2*split+1:3*split], X[3*split+1:4*split], X[4*split+1:5*split], X[5*split+1:] 
	mean1, mean2, mean3, mean4, mean5, mean6  = X1.mean(), X2.mean(), X3.mean(), X4.mean(), X5.mean(), X6.mean()
	var1, var2, var3, var4, var5, var6 = X1.var(), X2.var(), X3.var(), X4.var(), X5.var(), X6.var()
	st.write('mean1=%f, mean2=%f, mean3=%f, mean4=%f, mean5=%f, mean6=%f' % (mean1, mean2, mean3, mean4, mean5, mean6))
	st.write('variance1=%f, variance2=%f, variance3=%f, variance4=%f, variance5=%f, variance6=%f' % (var1, var2, var3, var4, var5, var6))

	st.header("METHOD 3: AUGMENTED DICKEY FULLER(ADF) TEST")
	result = adfuller(df_pollutant)
	st.write('ADF Statistic: %f' % result[0])
	st.write('p-value: %f' % result[1])
	st.write('Critical Values:')
	for key, value in result[4].items():
		st.write('\t%s: %.3f' % (key, value))
	if p <= 0.05:
		st.write("The data is Stationary and can be forecasted using a machine learning model.")
	else:
		st.write("The data is Non-Stationary and cannot be forecasted using a machine learning model.")

def stationarity_page():
	poll = st.sidebar.selectbox('Select the pollutant',('NO2','SO2', 'SPM', 'RSPM'))

	if poll == 'NO2':
		stationarity_check(df.no2)
	elif poll == 'SO2':
		stationarity_check(df.so2)
	elif poll == 'RSPM':
		stationarity_check(df.rspm)
	elif poll == 'SPM':
		stationarity_check(df.spm)



if __name__ == "__main__":
	main()