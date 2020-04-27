import streamlit as st
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import datetime


# Function to cache and load the dataset
@st.cache
def load_data():
    df = pd.read_csv('abc.csv', parse_dates=['date'], index_col = 'date')
    return df
# Loading the dataset
df = load_data()
aqi = pd.read_csv('aqi.csv', parse_dates=['date'], index_col = 'date')

def main():
    
    st.sidebar.title("INDEX")
    page = st.sidebar.selectbox("CONTENTS",["Introduction", "Location Wise", "Prediction", "Forecasting", "Historical Data", 'User Manual'])

    if page == "Introduction":
    	#st.title('Air Quality Index')
    	page1()
    
    elif page == "Prediction":
    	st.title('7 Day Prediction')
    	arima_pollutant()

    elif page == "Historical Data":
    	st.title("Historical")
    	historical()

    elif page == "Location Wise":
    	st.title('Location Specific AQI')
    	location()

    elif page == 'User Manual':
    	st.title('Air Pollution at an individual level')
    	image = Image.open('dos.jpg')
    	st.image(image,use_column_width = True)

#-------------------------------------------------------------------------------------------#
#--------------------------------------INTRODUCTION-----------------------------------------#
def page1():
	st.sidebar.header('AQI')
	page_1 = st.sidebar.radio("Select a page:",['Welcome', 'AQI_basics', 'How to use aqi'])

	if page_1 == 'Welcome':
		st.title('DELHI CITY - AIR NOW')
		image = Image.open('welcome.png')
		st.image(image,use_column_width = True)

	elif page_1 == 'AQI_basics':
		st.title('Air Quality Index - Basics')
		intro()

	elif page_1 == "How to use aqi":
		st.title('How to use AQI')
		htua()

def intro():

	# AQI intro
	st.subheader("How does AQI work?")
	'''Think of the AQI as a yardstick that runs from 0 to 500. The higher the AQI value, the greater the level of air
	pollution and the greater the health concern. For example, an AQI value of 50 or below represents good air quality, 
	while an AQI value over 300 represents hazardous air quality.   
	For each pollutant an AQI value of 100 generally corresponds to an ambient air concentration that equals the level of 
	the short-term national ambient air quality standard for protection of public health. AQI values at or below 100 are 
	generally thought of as satisfactory. When AQI values are above 100, air quality is unhealthy: at first for certain 
	sensitive groups of people, then for everyone as AQI values get higher.   
	The AQI is divided into six categories. Each category corresponds to a different level of health concern. 
	Each category also has a specific color. The color makes it easy for people to quickly determine whether air quality 
	is reaching unhealthy levels in their communities.'''

	# picture
	image = Image.open('aqi_table.jpg')
	st.image(image, caption = 'AQI Basics for Ozone and Particle Pollution',use_column_width = True)

	st.subheader('Major Pollutants')
	'''EPA establishes an AQI for major air pollutants regulated by the Clean Air Act. 
	Each of these pollutants has a national air quality standard set by EPA to protect public health:

	Particle Pollution (also known as particulate matter, including PM2.5 and PM10)
	Carbon Monoxide
	Sulfur Dioxide
	Nitrogen Dioxide'''

def htua():
	st.write('Use the Air Quality Index (AQI) to learn more about your local air quality and the best times for your outdoor activities.')

	st.subheader('Air Quality Index – daily index')

	'''**What it is:** The Air Quality Index, or AQI, is EPA’s tool for communicating daily air quality. It uses 
	color-coded categories and provides statements for each category that tell you about air quality in your area, 
	which groups of people may be affected, and steps you can take to reduce your exposure to air pollution. It’s also 
	used as the basis for air quality forecasts and current air quality reporting.'''  

	'''**Who issues it:** EPA has issued a national index for air quality since 1976 to provide an easy-to-understand 
	daily report on air quality in a format that’s the same from state to state. The AQI as we know it today was issued 
	in 1999; it’s been updated several times since to reflect the latest health-based air quality standards.'''  

	'''**What pollutants it covers:** Five major pollutants that are regulated by the Clean Air Act: ozone, 
	particle pollution (also called particulate matter), carbon monoxide, nitrogen dioxide and sulfur dioxide. 
	The AQI for each pollutant is generally based on the health-based national ambient air quality standard for that 
	pollutant and the scientific information that supports that standard.'''  

	'''**What time frame it covers:** It varies by pollutant. The ozone AQI is an 8-hour index; for particle pollution, 
	it’s 24 hours.'''  

	'''**Where can you get it:** Metro areas with a population of more than 350,000 are required to report the daily AQI. 
	Many more areas report it as a public service. You can find the daily AQI on AirNow and on state and local agency
	 websites. Some agencies also report the AQI via their local news media, or by telephone hotlines.'''  

	'''**How to use it:** Check the previous day’s AQI to learn more about air quality in your community.'''

	st.subheader('AQI Forecasts')

	'''**What they are:** A prediction of the day’s AQI. Forecasts usually are issued in the afternoon for the next day.'''

	'''**Who issues them:** State and local air quality forecasters across the country. They use a number of tools – 
	including weather forecast models, satellite images, air monitoring data, and computer models that estimate how 
	pollution travels on the air. They also use their own knowledge of how pollution behaves in certain communities to 
	issue the air quality forecast for the next day.'''  

	'''**What pollutants they cover:** Most state and local air quality forecasters issue forecasts for ozone and particle pollution,
	which are two of the most widespread pollutants in the U.S. A few areas also issue forecasts for nitrogen dioxide and 
	carbon monoxide.'''  

	'''**What time frame they cover:** In most areas, AQI forecasts focus on the next day. For ozone, an AQI forecast focuses on 
	the period during the day when average 8-hour ozone concentrations are expected to be the highest. For PM, the forecast
	 predicts the average 24-hour concentration for the next day.'''  

	'''**What they tell you:** AQI forecasts tell you what the next day’s AQI is expected to be, which groups of people may be 
	affected, and steps individuals can take to reduce their exposure to air pollution.'''  

	'''**Where you can get them:** State and local agencies provide AQI forecasts as a public service. You can find forecasts 
	for your area on AirNow, on state, local and tribal air agency websites, in your local news media, and through some
	 national media outlets.'''  

	'''**How to use them:** Use AQI forecasts to help you plan your outdoor activities for the day. Much like a weather forecast 
	lets you know whether to pack an umbrella, an air quality forecast lets you know when you may want to change your outdoor
	activities to reduce the amount of air pollution you breathe in. Many forecasters also provide a “forecast discussion,”
	which lets you know when pollution is expected to be highest during the day – and if there are times when air quality
	is expected to be better.'''  

	'''Some areas issue AQI forecasts for several days, to help you plan. But because things can change, it’s a good idea to
	check the forecast every day.'''


#-------------------------------------------------------------------------------------------#
#----------------------------------------ARIMA PAGE-----------------------------------------#

def arima(pol_df,p,d,q,plot_poll):
	train = pol_df[:'2012-12-31']

	# Model 
	model = ARIMA(train, order = (p,d,q))
	results = model.fit()
	#st.write(results.summary())

	# prediction
	model_predictions, SE, interval = results.forecast(steps = 7, alpha = 0.05)

	prediction_df = pd.DataFrame(columns=['Predictions', 'Minimum', 'Maximum'], index=pol_df.index[3288:3295])
	prediction_df['Predictions'] = model_predictions

	for i in range(0, 7):
	    prediction_df['Minimum'][i] = interval[i][0]
	    prediction_df['Maximum'][i] = interval[i][1]

	st.table(prediction_df)

def arima_pollutant():
	pol = st.sidebar.radio("Select the pollutant",["no2","so2","rspm","spm"])

	if pol == 'no2':
		dfcopy = df.copy()
		df_no2 = pd.DataFrame()
		df_no2 = dfcopy.drop(['so2','rspm','spm'], axis=1)
		st.subheader('Nitrogen Dioxide')
		arima(df_no2,1,0,1,df['no2'])

	elif pol == 'so2':
		dfcopy = df.copy()
		df_so2 = pd.DataFrame()
		df_so2 = dfcopy.drop(['no2','rspm','spm'], axis=1)
		st.subheader('Sulfur Dioxide')
		arima(df_so2,1,0,1,df['so2'])

	elif pol == 'spm':
		dfcopy = df.copy()
		df_spm = pd.DataFrame()
		df_spm = dfcopy.drop(['so2','rspm','no2'], axis=1)
		st.subheader('Suspended Particulate Matter')
		arima(df_spm,1,0,1,df['spm'])

	elif pol == 'rspm':
		dfcopy = df.copy()
		df_rspm = pd.DataFrame()
		df_rspm = dfcopy.drop(['so2','no2','spm'], axis=1)
		st.subheader('Residual Suspended Particulate Matter')
		arima(df_rspm,1,0,1,df['rspm'])

#-------------------------------------------------------------------------------------------#
#-------------------------------------HISTORICAL DATA---------------------------------------#

def historical():
	d = st.sidebar.date_input('Pollutant concentration for the date')
	if datetime.date(2004, 1, 1) <= d <= datetime.date(2015, 12, 12):
		st.write(aqi.loc[d])
	else:
		st.warning('Enter date between January 2004 to December 2015')

#-------------------------------------------------------------------------------------------#
#-------------------------------------HISTORICAL DATA---------------------------------------#

loc_aqi = pd.read_csv('location_aqi.csv', parse_dates=['date'], index_col = 'date')

def location():
	location = ['Mayapuri', 'N. Y. School', 'Siri Fort', 'Janakpuri', 'Nizamuddin', 'Shahzada Bagh', 'Pritampura', 'Shahdara', 'Ashok Vihar', 'BSZ Marg', 'DCE']
	loc = st.sidebar.selectbox('Choose a location', location)

	for i in range(0,len(location)):
		if loc == location[i]:
			data = loc_aqi.loc[loc_aqi['location_monitoring_station'] == location[i]]
			line_plot(data)
		else:
			continue

def line_plot(df1):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df1.index,y=df1['aqi'],name="AQI",line_color='red',opacity=0.8))
	fig.update_layout(xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

	st.table(df1)

#-------------------------------------------------------------------------------------------#
#-------------------------------------HISTORICAL DATA---------------------------------------#



#-------------------------------------------------------------------------------------------#
#------------------------------------------MAIN---------------------------------------------#
if __name__ == "__main__":
	main()