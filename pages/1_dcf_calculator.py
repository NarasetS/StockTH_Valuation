import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

st.header('1_DCF_Calculation')
############################ Side Bar #####################################################################
ticker = st.sidebar.text_input(
    "Ticker from Yahoo Finance : ", value='mc.bk' , placeholder= "Type a ticker..."
)
data = yf.Ticker(ticker)

sector = data.info['sector']
st.sidebar.markdown(
    "Sector : " + data.info['sector']
)

shares = data.info['sharesOutstanding']
input_shares = st.sidebar.number_input(
    "Shares Outstanding : ", value=shares , placeholder= "Input shares outstanding..."
)

## Net Debt = Long Term Borrowings - Cash ###
if sector == 'Financial Services':
    totalnoncurrentliabilities = data.balancesheet.loc[data.balancesheet.index == 'Net Debt'].values[0][0]
else:
    totalnoncurrentliabilities = data.balancesheet.loc[data.balancesheet.index == 'Total Non Current Liabilities Net Minority Interest'].values[0][0] - data.balancesheet.loc[data.balancesheet.index == 'Cash And Cash Equivalents'].values[0][0]
input_totalnoncurrentliabilities = st.sidebar.number_input(
    "total non-current liabilities : ", value=totalnoncurrentliabilities , placeholder= "Input total non-current liabilities.."
)

## Average Free Cash Flow ## 
year = 4
if sector == 'Financial Services':
    averagefreecashflow = (data.cash_flow.T['Free Cash Flow'] + data.cash_flow.T['Repayment Of Debt']).dropna()[:year].mean()
    # print((data.cash_flow.T['Free Cash Flow'] + data.cash_flow.T['Repayment Of Debt']).dropna()[:year])
else:
    averagefreecashflow = data.cash_flow.T['Free Cash Flow'].dropna()[:year].mean()
    # print(data.cash_flow.T['Free Cash Flow'].dropna()[:year])
input_averagefreecashflow = st.sidebar.number_input(
    "average freecashflow : ", value=averagefreecashflow , placeholder= "Input average freecashflow.."
)

input_year_growth = int(st.sidebar.number_input(
    "year_growth : ", value=3 , placeholder= "Input.."
))

input_year_sustain = int(st.sidebar.number_input(
    "year_sustain : ", value=7 , placeholder= "Input.."
))

input_growthrate = st.sidebar.number_input(
    "Expected Growth Rate for FCF : ", value=0.1 , placeholder= "Input.."
)

input_growth_to_sustain_ratio = st.sidebar.number_input(
    "growth_to_sustain_ratio : ", value=0.5 , placeholder= "Input.."
)

input_terminalgrowthrate = st.sidebar.number_input(
    "Terminal Rate for FCF : ", value=0.03 , placeholder= "Input.."
)

input_discountrate = st.sidebar.number_input(
    "Discount Rate : ", value=0.10 , placeholder= "Input.."
)

############################ Side Bar ################################################################



## DCF Calculation Function ## 
def intrinsic_value(
        shares,
        totalnoncurrentliabilities,
        averagefreecashflow,
        year_growth,
        year_sustain,
        growth_to_sustain_ratio,
        dr,
        tgr,
        gr
        ):
    
    PV_1 = [averagefreecashflow * ((1+gr)**i) / ((1+dr)**i) for i in range(1,year_growth+1,1)]
    PV_2 = [averagefreecashflow * ((1+gr)**year_growth) * ((1+(gr*growth_to_sustain_ratio))**i)/((1+dr)**(year_growth+i)) for i in range(1,year_sustain+1,1)]
    TV = averagefreecashflow * ((1+gr)**year_growth) * ((1+(gr*growth_to_sustain_ratio))**(year_sustain)) * (1+tgr) / ((dr-tgr)*((1+dr)**(year_growth+year_sustain)))
    return (np.sum(PV_1) + np.sum(PV_2) + TV - totalnoncurrentliabilities)/shares
## DCF Calculation Function ##

## Reverse DCF Function ##
def intrinsic_value_diff_currentprice(
        shares,
        totalnoncurrentliabilities,
        averagefreecashflow,
        year_growth,
        year_sustain,
        growth_to_sustain_ratio,
        dr,
        tgr,
        currentprice,
        gr
        ):
    
    PV_1 = [averagefreecashflow * ((1+gr)**i) / ((1+dr)**i) for i in range(1,year_growth+1,1)]
    PV_2 = [averagefreecashflow * ((1+gr)**year_growth) * ((1+(gr*growth_to_sustain_ratio))**i)/((1+dr)**(year_growth+i)) for i in range(1,year_sustain+1,1)]
    TV = averagefreecashflow * ((1+gr)**year_growth) * ((1+(gr*growth_to_sustain_ratio))**(year_sustain)) * (1+tgr) / ((dr-tgr)*((1+dr)**(year_growth+year_sustain)))
    # print((((np.sum(PV_1) + np.sum(PV_2) + TV - totalnoncurrentliabilities)/shares)) , currentprice)
    return (((np.sum(PV_1) + np.sum(PV_2) + TV - totalnoncurrentliabilities)/shares) - currentprice)/currentprice

def newton_raphson_method(
            shares,
            totalnoncurrentliabilities,
            averagefreecashflow,
            year_growth,
            year_sustain,
            growth_to_sustain_ratio,
            dr,
            tgr,
            currentprice):
    
    initial_guess = 0.01
    tolerance = 0.01
    max_iterations = 1000000
    x_n = initial_guess
    
    for iteration in range(max_iterations):
    # while (True):
        f_x = intrinsic_value_diff_currentprice(
            shares,
            totalnoncurrentliabilities,
            averagefreecashflow,
            year_growth,
            year_sustain,
            growth_to_sustain_ratio,
            dr,
            tgr,
            currentprice,
            x_n
        )

        if abs(f_x) < tolerance :
            break

        x_n = x_n - f_x * 0.01

    return x_n
## Reverse DCF Function ##


#### Main area #############################################################################################
st.markdown('To perform discounted cash flow to find intrinsic value for : ' + ticker)
iv = np.round(intrinsic_value(
        input_shares,
        input_totalnoncurrentliabilities,
        input_averagefreecashflow,
        input_year_growth,
        input_year_sustain,
        input_growth_to_sustain_ratio,
        input_discountrate,
        input_terminalgrowthrate,
        input_growthrate
        ),2)
st.markdown('intrinsic value = ' + str(iv))
currentprice = data.info['currentPrice']
st.markdown('current price : ' + str(currentprice))
st.markdown('margin of safety (%) : ' + str(
    np.round((data.info['currentPrice'] - iv)*100/data.info['currentPrice'],2)
))

try:
    st.markdown('profitMargins : ' + str(data.info['profitMargins']))
    st.markdown('returnOnEquity : ' + str(data.info['returnOnEquity']))
    st.markdown('trailingPE : ' + str(data.info['trailingPE']))
    st.markdown('priceToBook : ' + str(data.info['priceToBook']))
    st.markdown('debtToEquity (%) : ' + str(data.info['debtToEquity']))
    st.markdown('currentRatio : ' + str(data.info['currentRatio']))
    st.markdown('totalCashPerShare : ' + str(data.info['totalCashPerShare']))
    st.markdown('pegRatio : ' + str(data.info['pegRatio']))
    
except:
    st.markdown('Theres error loading indicative ratios')


st.header('Reverse DCF Curve (Expected of FCF vs Discount Rate) based on current price')
st.markdown('To indicate expected growth rate based on particular discount rate which makes intrinsic value calculated from DCF equal to current price..')

## Perform Reverse DCF ##
dr_list = [dr*0.01 for dr in range(int((input_terminalgrowthrate*100)+1),101,1)]
df = pd.DataFrame(index=dr_list,columns=['growthrate'])
growthrate = [ newton_raphson_method(           
            input_shares,
            input_totalnoncurrentliabilities,
            input_averagefreecashflow,
            input_year_growth,
            input_year_sustain,
            input_growth_to_sustain_ratio,
            dr,
            input_terminalgrowthrate,
            currentprice) for dr in dr_list]
df['growthrate'] = growthrate
df.reset_index(inplace=True)
df = df.rename(columns={'index':'discountrate'})
st.scatter_chart(df, x = 'discountrate', y = 'growthrate')
## Perform Reverse DCF ##

## test ####
# root = newton_raphson_method(           
#             input_shares,
#             input_totalnoncurrentliabilities,
#             input_averagefreecashflow,
#             input_year_growth,
#             input_year_sustain,
#             input_growth_to_sustain_ratio,
#             input_discountrate,
#             input_terminalgrowthrate,
#             currentprice)
# st.markdown(root)
# diff = intrinsic_value_diff_currentprice(
#         input_shares,
#         input_totalnoncurrentliabilities,
#         input_averagefreecashflow,
#         input_year_growth,
#         input_year_sustain,
#         input_growth_to_sustain_ratio,
#         input_discountrate,
#         input_terminalgrowthrate,
#         currentprice,
#         root
#         ) * 100
# st.markdown(diff)
## test ####

#### Main area #############################################################################################0