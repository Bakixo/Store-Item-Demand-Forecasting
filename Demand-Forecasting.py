# Dataset in here --> https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
# !pip install lightgbm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("################# SHAPE #################")
    print(dataframe.shape)
    print("################# TYPES #################")
    print(dataframe.dtypes)
    print("################# HEAD #################")
    print(dataframe.head(head))
    print("################# TAİL #################")
    print(dataframe.tail(head))
    print("################# NA #################")
    print(dataframe.isna().sum())
    print("################# QUANTİLES #################")
    # Sadece sayısal sütunların kuantillerini hesaplayın
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    print(dataframe[numeric_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



# <> IMPORT DATA <>

train = pd.read_csv("train.csv",parse_dates=['date'])
test = pd.read_csv("test.csv",parse_dates=['date'])

sample_sub = pd.read_csv("sample_submission.csv")


df = pd.concat([train,test],sort=False)

# <> Exploratory Data Analysis <>

firstd = df["date"].min()
lastd = df["date"].max()

#check_df(df)

# print(df["store"].nunique()) 10 adet mağaza var. 
# print(df["item"].nunique()) 50 adet ürün var.


# print(df.groupby(["store"])["item"].nunique())

df.groupby(["store","item"]).agg({"sales" : ["sum"]})

df.groupby(["store","item"]).agg({"sales" : ["sum","median","mean","std"]})


# <> Feature Engineering <>


def create_date_features(df):
    df["month"] = df.date.dt.month
    df["day_of_month"] = df.date.dt.day
    df["day_of_year"] = df.date.dt.dayofyear
    df["week_of_year"] = df.date.dt.isocalendar().week
    df["day_of_week"] = df.date.dt.dayofweek
    df["year"] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df["is_month_start"] = df.date.dt.is_month_start.astype(int)
    df["is_month_end"] = df.date.dt.is_month_end.astype(int)

    return df

df = create_date_features(df)

df.groupby(["store","item","month"]).agg({"sales" : ["sum","median","mean","std"]})


# <> RANDOM NOİSE <>

def random_noise(df):
    return np.random.normal(scale=1.6,size=(len(df),)) # aşırı öğrenmenin önüne geçmek için.


# <> Lag/Shifted features <>

df.sort_values(by=["store","item","date"],axis=0, inplace=True)

pd.DataFrame({"sales": df["sales"].values[0:10],
            "lag1": df["sales"].shift(1).values[0:10],
            "lag2": df["sales"].shift(2).values[0:10],
            "lag3": df["sales"].shift(3).values[0:10],
            "lag4": df["sales"].shift(4).values[0:10],})

#print(df.groupby(["store","item"])["sales"].transform(lambda x: x.shift(1)))

def lag_features(df,lags):
    for lag in lags:
        df["sales_lag" + str(lag)] = df.groupby(["store","item"])["sales"].transform(
            lambda x : x.shift(lag)) + random_noise(df) 
    return df

df = lag_features(df, [91,98,105,112,119,126,182,364,546,728])



# <> Rolling mean Features <>

pd.DataFrame({"sales": df["sales"].values[0:10],
            "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
            "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
            "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10],})

def roll_mean_features(df,windows):
    for window in windows:
        df["sales_roll_mean" + str(window)] = df.groupby(["store","item"])["sales"]. \
                                                transform(
            lambda x : x.shift(1).rolling(window=window,min_periods = 10, win_type = "triang").mean()) + random_noise(df) 
    return df



df = roll_mean_features(df, [365, 546])


# <> Exponentially weighted mean features <> 

pd.DataFrame({"sales": df["sales"].values[0:10],
            "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
            "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10], 
            "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
            "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
            "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})


def ewm_features(df,alphas,lags):
    for alpha in alphas:
        for lag in lags:
            df['sales_ewm_alpha_' + str(alpha).replace(".","") + "_lag_" + str(lag)] = \
                df.groupby(["store","item"])["sales"].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return df


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]


df = ewm_features(df,alphas=alphas,lags=lags)



# <> ONE-HOT Encoding <>

df = pd.get_dummies(df, columns= ['store','item','day_of_week','month'])

                 # buralarda kod şüpheli !!! 


# <> Bağımlı değişkenin logaritması (converting sales to log(1 + sales)) <>

df["sales"] = np.log1p(df["sales"].values)

            # bu satıra gerek yok ama train işini hızlı yapsın diye numeric sütunun logaritmasını aldık.



# <> Custom Cost Function <>
# MAE, MSE, RMSE, SSE
# MAPE
# !SMAPE : Symmetric mean absoulute percentage error (adjusted MAPE)


def smape(preds,target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val               
    # SMAPE nin algoritması 


def lgbm_smape(preds, train_data): 
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val,False
    # lightgbm için smape



# <> Time-Based Validation Sets <> 

    # print(train,test)

    # 2017'nin başına kadar (2016'nın sonuna kadar) train seti 
train = df.loc[(df["date"] < "2017-01-01"), :]
    # 2017'nin ilk üç ayının validasyon seti 
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]  # test seti 

cols = [col for col in train.columns if col not in ["date", "id", "sales", "year"]] # bağımsızlar sütunu

Y_train = train["sales"]
X_train = train[cols]

y_val = val["sales"]
x_val = val[cols]



# <> lightGBM parameters <>

lgb_params = {'num_leaves' : 10,
              'learning_rate' : 0.02,
              'feature_fraction' : 0.8,
              'max_depth' : 5,
              'verbose': 0,
              'num_boost_round': 1000, ## ÖNEMLİ
              'early_stopping_rounds' : 200,
              'nthread' : -1 }
            # normal de deneme yanılma yoluyla sayıları bulmamız gerekiyor ben bu veri için hazır aldım 
            
        # num_leaves : bir ağaçtaki maksimum yaprak sayısı
        # learning_rate : shrinkage_rate, eta şeklinde de karşımıza çıkabilir. Öğrenme oranı.
        # feature_fraction : random forestin subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayisi.    
        # max_depth : maksimum derinlik.
        # num_boost_round : n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.
        # early_stopping_rounds : validasyon setindeki metrik belirli bir easy_stopping_rounds da ilerlemiyorsa yani
    #hata düşmüyorsa modellemeyi durdurur. hem train süresini kısaltır hemde overfitting e engel olur.
         # nthread : işlemcinin tam performans kullanılmasını ifade ediyor.


lgbtrain = lgb.Dataset(data=X_train,label=Y_train,feature_name=cols) # lgb daha hızlı çalışsın diye kendi datasetini oluşturduk.

lgbval = lgb.Dataset(data=x_val, label=y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(
    lgb_params,
    lgbtrain,
    valid_sets=[lgbtrain, lgbval],
    num_boost_round=lgb_params['num_boost_round'],
    feval=lgbm_smape,
    callbacks=[
        lgb.early_stopping(stopping_rounds=lgb_params['early_stopping_rounds']),
        lgb.log_evaluation(period=100)  # Burada 'verbose_eval' yerine 'log_evaluation' kullanıyoruz
    ]
)


#y_pred_val = model.predict(x_val,num_iteration=model.best_iteration)

#smape(np.expm1(y_pred_val), np.expm1(y_val)) # 13.3451



# <> Değişken önem düzeyleri <>

def plot_lgb_importance(model,plot=False,num=10):

    gain = model.feature_importance("gain")
    feat_imp = pd.DataFrame({'feature' : model.feature_name(),
                            'split' : model.feature_importance('split'),
                            'gain' : 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    
    if plot:
        plt.figure(figsize=(10,10))
        sns.set(font_scale=1)
        sns.barplot(x='gain', y='feature', data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


#plot_lgb_importance(model=model, num=30) 

#plot_lgb_importance(model, plot=True, num=30) # değişkenler arasın önem düzeyi grafiği


# <> final model <>
    
    
    
        # extra 
"""

feat_imp = plot_lgb_importances(model, num=200)
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

[col for col in cols if col not in importance_zero]

diyerek boş elemanları çıkarmış halini de cols yerine alabiliriz bu daha iyi de yapabilir.
"""

train = df.loc[~df.sales.isna()]
Y_train = train["sales"]
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves' : 10,
              'learning_rate' : 0.02,
              'feature_fraction' : 0.8,
              'max_depth' : 5,
              'verbose': 0,
              'num_boost_round': model.best_iteration, ## ÖNEMLİ
              'nthread' : -1 }

lgbtrain_all = lgb.Dataset(data=X_train,label=Y_train, feature_name=cols)

Final_Model = lgb.train(lgb_params,lgbtrain_all, num_boost_round=model.best_iteration)

test_pred = Final_Model.predict(X_test, num_iteration=model.best_iteration)

# 2.56074216 2.7395261  2.70628238 ... 4.36618308 4.41422348 4.48623857] # logaritması alınmış değerler


# <> submission File <>


submission_df = test.loc[:,["id","sales"]]
submission_df["sales"] = np.expm1(test_pred)

submission_df["id"] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)