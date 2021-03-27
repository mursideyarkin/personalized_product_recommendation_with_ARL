from mlxtend.frequent_patterns import apriori, association_rules


def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def crm_data_prep(dataframe):
    dataframe.dropna(axis=0, inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    return dataframe


def create_invoice_product_df(dataframe):
    return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)


def create_cltv_p(dataframe):
    import datetime as dt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from lifetimes import BetaGeoFitter
    from lifetimes import GammaGammaFitter

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




def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
    df['title'] = df['title'].apply(lambda x: x.strip())
    a = pd.DataFrame(df["title"].value_counts())
    rare_movies = a[a["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


def user_based_recommender():
    import pickle
    import pandas as pd
    user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))
    random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
    random_user_df = user_movie_df[user_movie_df.index == random_user]

    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                          random_user_df[movies_watched]])
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
    temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
    temp.columns = ['sum_corr', 'sum_weighted_rating']

    recommendation_df = pd.DataFrame()
    recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
    recommendation_df['movieId'] = temp.index
    recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)

    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'])]