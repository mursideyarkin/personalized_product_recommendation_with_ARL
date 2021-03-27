######################################################################
# PERSONALIZED PRODUCT RECOMMENDATIONS WITH ASSOCIATION RULE LEARNING
######################################################################

# PROJECT STEPS:

# 1. Importing data set and libraries
# 2. Data set check
# 3. Data preprocessing
# 4. 6-months customer lifetime value prediction and customer segmentation
# 5. Creating new data sets according to segments
# 6. Generation of product association rules for each segment
# 7. Product recommendations for German customers

##########################################
# STEP 1: IMPORT DATA SETS AND LIBRARIES
##########################################

import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

# Importing data from external file
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

######################################
# STEP 2: DATA SET CHECK
######################################

from helpers import check_df
check_df(df)

# There are null observations in the dataset, especially in the Customer ID column.
# Since this analysis will be consumer-based, we need to remove these observations from the dataset.
# There are negative values in the Quantity and Price columns.These transactions are canceled orders.
# In the next step (data preprocessing), we'll eliminate these observations.

######################################
# STEP 3: DATA PREPROCESSING
######################################

from helpers import crm_data_prep
# Data preprocessing steps in imported function:
# 1. Removing null oberservations and canceled orders
# 2. Replacing outliers in the Quantity and Price columns with the upper limit
# 3. Calculating total price per transaction

# Execute function:
df_prep = crm_data_prep(df)

# Data set check after preprocessing:
check_df(df_prep)

#####################################################
# STEP 4: 6-MONTHS CLTV PREDICTION AND SEGMENTATION
#####################################################

# Defining function:
def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)

    ## recency kullanıcıya özel dinamik.
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max()-date.min()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = rfm.columns.droplevel(0)

    ## recency_cltv_p
    rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

    ## basitleştirilmiş monetary_avg
    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

    rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)


    # BGNBD için WEEKLY RECENCY VE WEEKLY T'nin HESAPLANMASI
    ## recency_weekly_cltv_p
    rfm["recency_weekly_cltv_p"] = rfm["recency_cltv_p"] / 7
    rfm["T_weekly"] = rfm["T"] / 7



    # KONTROL
    rfm = rfm[rfm["monetary_avg"] > 0]

    ## recency filtre (daha saglıklı cltvp hesabı için)
    rfm = rfm[(rfm['frequency'] > 1)]

    rfm["frequency"] = rfm["frequency"].astype(int)

    # BGNBD
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly_cltv_p'],
            rfm['T_weekly'])

    # exp_sales_1_month
    rfm["exp_sales_1_month"] = bgf.predict(4,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])
    # exp_sales_3_month
    rfm["exp_sales_3_month"] = bgf.predict(12,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])

    # expected_average_profit
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                             rfm['monetary_avg'])
    # 6 aylık cltv_p
    cltv = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly_cltv_p'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)

    rfm["cltv_p"] = cltv

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p"]])
    rfm["cltv_p"] = scaler.transform(rfm[["cltv_p"]])

    # rfm.fillna(0, inplace=True)

    # cltv_p_segment
    rfm["cltv_p_segment"] = pd.qcut(rfm["cltv_p"], 3, labels=["C", "B", "A"])

    ## recency_cltv_p, recency_weekly_cltv_p
    rfm = rfm[["recency_cltv_p", "T", "monetary_avg", "recency_weekly_cltv_p", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit",
               "cltv_p", "cltv_p_segment"]]


    return rfm

# CLTV prediction and segmentation:
cltv_p = create_cltv_p(df_prep)
check_df(cltv_p)
cltv_p.head()

#####################################################
# STEP 5: CREATING NEW DATA SETS ACCORDING TO SEGMENTS
#####################################################

# Getting customer id's for each segment:
a_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "A"].index
b_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "B"].index
c_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "C"].index

# Creating new data sets according to segments
a_segment_df = df_prep[df_prep["Customer ID"].isin(a_segment_ids)]
b_segment_df = df_prep[df_prep["Customer ID"].isin(b_segment_ids)]
c_segment_df = df_prep[df_prep["Customer ID"].isin(c_segment_ids)]

#####################################################
# STEP 6: PRODUCT ASSOCIATION RULES FOR EACH SEGMENT
#####################################################

# Importing function for product matrix
from helpers import create_invoice_product_df

# Defining function for creating rules
def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.02)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.02)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules

# Defining rule and selecting product according to lift ratio for segment A:
rules_a = create_rules(a_segment_df)
product_a = (rules_a["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

# Defining rule and selecting product according to lift ratio for segment B:
rules_b = create_rules(b_segment_df)
product_b = int(rules_b["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

# Defining rule and selecting product according to lift ratio for segment C:
rules_c = create_rules(c_segment_df)
product_c = int(rules_c["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

# Defining function for getting product name:
def check_id(stock_code):
    product_name = df_prep[df_prep["StockCode"] == stock_code][["Description"]].values[0].tolist()
    return print(product_name)

# Recommended product's name:
check_id(product_a)
check_id(product_b)
check_id(product_c)

##############################################################
# STEP 7: PRODUCT RECOMMENDATIONS FOR CUSTOMERS FROM GERMANY
##############################################################

# Creating new empty variable to store recommendations:
cltv_p["recommended_product"] = " "

# Getting customers id's who is from Germany
germany_ids = df_prep[df_prep["Country"] == "Germany"]["Customer ID"].drop_duplicates()

# Recommending product for each segment:
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A"), "recommended_product"] = product_a
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "B"), "recommended_product"] = product_b
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "C"), "recommended_product"] = product_c

# Number of recommended products:
cltv_p[cltv_p.index.isin(germany_ids)]["recommended_product"].value_counts()