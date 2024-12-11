# import streamlit as st
# import pandas as pd
# import yfinance as yf
# from ta.volatility import BollingerBands
# from ta.trend import MACD, EMAIndicator, SMAIndicator
# from ta.momentum import RSIIndicator
# import datetime
# from datetime import date
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.metrics import r2_score, mean_absolute_error



# st.title('Stock Price Predictions')
# st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

# def main():
#     option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
#     if option == 'Visualize':
#         tech_indicators()
#     elif option == 'Recent Data':
#         dataframe()
#     else:
#         predict()



# @st.cache_resource
# def download_data(op, start_date, end_date):
#     df = yf.download(op, start=start_date, end=end_date, progress=False)
#     return df



# option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
# option = option.upper()
# today = datetime.date.today()
# duration = st.sidebar.number_input('Enter the duration', value=3000)
# before = today - datetime.timedelta(days=duration)
# start_date = st.sidebar.date_input('Start Date', value=before)
# end_date = st.sidebar.date_input('End date', today)
# if st.sidebar.button('Send'):
#     if start_date < end_date:
#         st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
#         download_data(option, start_date, end_date)
#     else:
#         st.sidebar.error('Error: End date must fall after start date')




# data = download_data(option, start_date, end_date)
# scaler = StandardScaler()

# def tech_indicators():
#     st.header('Technical Indicators')
#     option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

#     # Bollinger bands
#     bb_indicator = BollingerBands(data.Close)
#     bb = data
#     bb['bb_h'] = bb_indicator.bollinger_hband()
#     bb['bb_l'] = bb_indicator.bollinger_lband()
#     # Creating a new dataframe
#     bb = bb[['Close', 'bb_h', 'bb_l']]
#     # MACD
#     macd = MACD(data.Close).macd()
#     # RSI
#     rsi = RSIIndicator(data.Close).rsi()
#     # SMA
#     sma = SMAIndicator(data.Close, window=14).sma_indicator()
#     # EMA
#     ema = EMAIndicator(data.Close).ema_indicator()

#     if option == 'Close':
#         st.write('Close Price')
#         st.line_chart(data.Close)
#     elif option == 'BB':
#         st.write('BollingerBands')
#         st.line_chart(bb)
#     elif option == 'MACD':
#         st.write('Moving Average Convergence Divergence')
#         st.line_chart(macd)
#     elif option == 'RSI':
#         st.write('Relative Strength Indicator')
#         st.line_chart(rsi)
#     elif option == 'SMA':
#         st.write('Simple Moving Average')
#         st.line_chart(sma)
#     else:
#         st.write('Expoenetial Moving Average')
#         st.line_chart(ema)


# def dataframe():
#     st.header('Recent Data')
#     st.dataframe(data.tail(10))



# def predict():
#     model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
#     num = st.number_input('How many days forecast?', value=5)
#     num = int(num)
#     if st.button('Predict'):
#         if model == 'LinearRegression':
#             engine = LinearRegression()
#             model_engine(engine, num)
#         elif model == 'RandomForestRegressor':
#             engine = RandomForestRegressor()
#             model_engine(engine, num)
#         elif model == 'ExtraTreesRegressor':
#             engine = ExtraTreesRegressor()
#             model_engine(engine, num)
#         elif model == 'KNeighborsRegressor':
#             engine = KNeighborsRegressor()
#             model_engine(engine, num)
#         else:
#             engine = XGBRegressor()
#             model_engine(engine, num)


# def model_engine(model, num):
#     # getting only the closing price
#     df = data[['Close']]
#     # shifting the closing price based on number of days forecast
#     df['preds'] = data.Close.shift(-num)
#     # scaling the data
#     x = df.drop(['preds'], axis=1).values
#     x = scaler.fit_transform(x)
#     # storing the last num_days data
#     x_forecast = x[-num:]
#     # selecting the required values for training
#     x = x[:-num]
#     # getting the preds column
#     y = df.preds.values
#     # selecting the required values for training
#     y = y[:-num]

#     #spliting the data
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
#     # training the model
#     model.fit(x_train, y_train)
#     preds = model.predict(x_test)
#     st.text(f'r2_score: {r2_score(y_test, preds)} \
#             \nMAE: {mean_absolute_error(y_test, preds)}')
#     # predicting stock price based on the number of days
#     forecast_pred = model.predict(x_forecast)
#     day = 1
#     for i in forecast_pred:
#         st.text(f'Day {day}: {i}')
#         day += 1


# if __name__ == '__main__':
#     main()
# Page configuration must be the first Streamlit command

import streamlit as st
st.set_page_config(
    page_title="Stock Price Predictions",
    page_icon="📈",
    layout="wide"
)

import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Custom CSS with beautiful colors and gradients
st.markdown("""
<style>
/* Modern Color Palette with Enhanced Colors */
:root {
    --primary-color: #1A1F36;
    --secondary-color: #232B48;
    --accent-color: #4CAF50; /* Brightened for a more engaging look */
    --highlight-color: #16A085;
    --text-primary: #EAECEE;
    --text-secondary: #BDC3C7;
    --success-color: #27AE60;
    --warning-color: #F39C12;
    --danger-color: #C0392B;
    --card-bg: rgba(34, 47, 62, 0.8);
    --gradient-start: #34495E;
    --gradient-end: #1ABC9C;
}

/* Global Background with Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

/* Typography with Subtle Glow */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary);
    font-weight: 600;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.15);
    letter-spacing: 0.6px;
}

/* Paragraphs and Text */
p, span, div {
    color: var(--text-secondary);
    line-height: 1.6;
}

/* Sidebar Styling with Slide-in Animation */
[data-testid="stSidebar"] {
    background-color: rgba(34, 47, 62, 0.95);
    border-right: 1px solid rgba(26, 188, 156, 0.4);
    backdrop-filter: blur(10px);
    padding: 2rem 1rem;
    animation: slideInLeft 0.6s ease-out;
}

/* Title Container with Glassmorphism & Pulse */
.title-container {
    background: rgba(44, 62, 80, 0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    animation: pulse 1.5s infinite alternate;
}

/* Cards with Hover and Fade-up Animation */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: transform 0.4s ease, box-shadow 0.4s ease;
    animation: fadeUp 0.7s ease-out;
}

.glass-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
}

/* Metric Cards with Scale and Shadow Pulse */
.metric-card {
    background: linear-gradient(135deg, var(--highlight-color) 0%, var(--accent-color) 100%);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    animation: shadowPulse 2s infinite;
}

.metric-card:hover {
    transform: scale(1.08);
    box-shadow: 0 10px 20px rgba(26, 188, 156, 0.4);
}

/* Chart Container with Fade-in */
.chart-container {
    background: var(--card-bg);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.15);
    animation: fadeIn 1s ease-in-out;
}

/* Buttons with Gradient, Shadow & Hover Effect */
.stButton > button {
    background: linear-gradient(135deg, var(--highlight-color) 0%, var(--accent-color) 100%);
    color: var(--text-primary);
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.5rem;
    font-weight: 600;
    transition: transform 0.4s, box-shadow 0.4s ease;
    animation: popIn 0.5s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(26, 188, 156, 0.4);
    animation: buttonHover 0.3s ease;
}

/* Input Fields with Border Animation */
.stSelectbox > div > div, .stTextInput > div > div {
    background: rgba(44, 62, 80, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    color: var(--text-primary);
    transition: all 0.3s ease;
}

.stSelectbox > div > div:hover, .stTextInput > div > div:hover {
    border-color: var(--highlight-color);
}

/* Animated Progress Bar */
.stProgress > div > div > div {
    background-color: var(--accent-color);
    transition: width 0.4s ease;
}

/* Success and Error Messages with Slide-in Animation */
.success-message {
    background: rgba(39, 174, 96, 0.2);
    border: 1px solid var(--success-color);
    color: var(--success-color);
    padding: 1rem;
    border-radius: 8px;
    animation: slideIn 0.5s ease;
}

.error-message {
    background: rgba(192, 57, 43, 0.2);
    border: 1px solid var(--danger-color);
    color: var(--danger-color);
    padding: 1rem;
    border-radius: 8px;
    animation: slideIn 0.5s ease;
}

/* Animation Keyframes */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes shadowPulse {
    0%, 100% { box-shadow: 0 0 10px rgba(26, 188, 156, 0.3); }
    50% { box-shadow: 0 0 20px rgba(26, 188, 156, 0.5); }
}

@keyframes pulse {
    from { transform: scale(1); }
    to { transform: scale(1.02); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Button Hover Effect */
@keyframes buttonHover {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* Enhanced Placeholder Loading Animation */
.loading-placeholder {
    background: linear-gradient(90deg, rgba(255,255,255,0.1) 25%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0.1) 75%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite;
    border-radius: 8px;
    height: 100px;
    margin: 1rem 0;
}

/* Shimmer Effect */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

/* Scrollbar with New Color */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--primary-color);
}

::-webkit-scrollbar-thumb {
    background: var(--highlight-color);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-color);
}
</style>
""", unsafe_allow_html=True)

# Custom title with icon and animation
st.markdown("""
    <div class="title-container">
        <h1>
            <span style="font-size: 3rem; margin-right: 10px;">📈</span>
            Stock Price Predictions
        </h1>
        <p style="font-size: 1.2rem; color: #3498db; margin-top: 1rem;">
            Advanced Technical Analysis & Market Forecasting
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar styling with icons
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>
            <span style="font-size: 2rem; margin-right: 10px;">🎮</span>
            Control Panel
        </h2>
    </div>
""", unsafe_allow_html=True)

# Rest of your existing functions remain the same, just add icons to headers
@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

def main():
    option = st.sidebar.selectbox('📊 Select Analysis Mode', 
                                ['Visualize', 'Recent Data', 'Predict'],
                                help="Choose what type of analysis you want to perform")
    
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

# Enhanced sidebar inputs with icons
st.sidebar.markdown("---")
option = st.sidebar.text_input('🔍 Enter Stock Symbol', value='SPY')
option = option.upper()

today = datetime.date.today()
duration = st.sidebar.number_input('📅 Analysis Duration (days)', value=3000, min_value=1)
before = today - datetime.timedelta(days=duration)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input('📆 Start Date', value=before)
with col2:
    end_date = st.date_input('📆 End Date', today)

if st.sidebar.button('🚀 Analyze Stock'):
    if start_date < end_date:
        st.sidebar.success('✅ Analysis Period:\n\n📅 From: `%s`\n\n📅 To: `%s`' % (start_date, end_date))
        data = download_data(option, start_date, end_date)
    else:
        st.sidebar.error('❌ Error: End date must fall after start date')

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.markdown("<div class='stcard'>", unsafe_allow_html=True)
    st.header('📊 Technical Indicators')
    
    option = st.radio('Select Technical Indicator:', 
                     ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'],
                     horizontal=True)

    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb = bb[['Close', 'bb_h', 'bb_l']]
    
    macd = MACD(data.Close).macd()
    rsi = RSIIndicator(data.Close).rsi()
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    ema = EMAIndicator(data.Close).ema_indicator()

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    if option == 'Close':
        st.subheader('📈 Close Price Analysis')
        st.line_chart(data.Close, use_container_width=True)
    elif option == 'BB':
        st.subheader('📊 Bollinger Bands Analysis')
        st.line_chart(bb, use_container_width=True)
    elif option == 'MACD':
        st.subheader('📈 MACD Analysis')
        st.line_chart(macd, use_container_width=True)
    elif option == 'RSI':
        st.subheader('📊 RSI Analysis')
        st.line_chart(rsi, use_container_width=True)
    elif option == 'SMA':
        st.subheader('📈 SMA Analysis')
        st.line_chart(sma, use_container_width=True)
    else:
        st.subheader('📊 EMA Analysis')
        st.line_chart(ema, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def dataframe():
    st.markdown("<div class='stcard'>", unsafe_allow_html=True)
    st.header('📋 Recent Market Data')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>💰 Current Price</h3>
                <h2>${:.2f}</h2>
            </div>
        """.format(data.Close.iloc[-1]), unsafe_allow_html=True)
    
    with col2:
        daily_return = ((data.Close.iloc[-1] - data.Close.iloc[-2]) / data.Close.iloc[-2]) * 100
        st.markdown("""
            <div class="metric-card">
                <h3>📊 Daily Return</h3>
                <h2>{:.2f}%</h2>
            </div>
        """.format(daily_return), unsafe_allow_html=True)
    
    with col3:
        volume = data.Volume.iloc[-1]
        st.markdown("""
            <div class="metric-card">
                <h3>📈 Volume</h3>
                <h2>{:,.0f}</h2>
            </div>
        """.format(volume), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(data.tail(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def predict():
    st.markdown("<div class='stcard'>", unsafe_allow_html=True)
    st.header('🔮 Price Prediction')
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.radio('🤖 Select Prediction Model:', 
                        ['RandomForestRegressor', 
                         'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    with col2:
        num = st.number_input('📅 Forecast Horizon (Days)', value=5, min_value=1)
        num = int(num)
    
    if st.button('🎯 Generate Prediction'):
        with st.spinner('🔄 Calculating predictions...'):
            #if model == 'LinearRegression':
             #  engine = LinearRegression()
            if model == 'RandomForestRegressor':
                engine = RandomForestRegressor()
            elif model == 'ExtraTreesRegressor':
                engine = ExtraTreesRegressor()
            elif model == 'KNeighborsRegressor':
                engine = KNeighborsRegressor()
            else:
                engine = XGBRegressor()
            
            model_engine(engine, num)
    
    st.markdown("</div>", unsafe_allow_html=True)



def model_engine(model, num):
    # Model calculation logic
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    
    # Display results with enhanced styling
    st.markdown("""
    <div style='padding: 1rem; color: black; border-radius: 10px; margin-top: 1rem;'>
        <h3>Model Performance Metrics</h3>
    </div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
    with col2:
        st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, preds):.4f}")
    
    forecast_pred = model.predict(x_forecast)
    
    st.markdown("""
        <div style='padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h3>Price Forecasts</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Create a DataFrame for predictions
    forecast_df = pd.DataFrame({
        'Day': range(1, num + 1),
        'Predicted Price': forecast_pred
    })
    
    # Display predictions in a styled table
    st.dataframe(forecast_df, use_container_width=True)

if __name__ == '__main__':
    main()