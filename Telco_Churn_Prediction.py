#######################################
#Telco Churn Prediction
#######################################

#######################################
#İş Problemi
#######################################

#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

#########################################
#Veri Seti Hikayesi
#########################################

#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
#bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

#########################################
#Değişkenler
#########################################
#CustomerId : Müşteri İd’si
#Gender : Cinsiyet
#SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
#Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
#Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
#tenure : Müşterinin şirkette kaldığı ay sayısı
#PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
#MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
#InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
#OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
#StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
#Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
#PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
#PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
#MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
#TotalCharges : Müşteriden tahsil edilen toplam tutar
#Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır)
# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib
# pip install graphviz

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz
import warnings
warnings.simplefilter(action="ignore", category=Warning)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Telco_Churn_Prediction/Telco-Customer-Churn.csv")
df.head()
df.shape  # (7043, 21)

##############################################################
#Görev 1 : Keşifçi Veri Analizi
##############################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

#########
#Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
#########

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
df[cat_cols].head()

num_cols
df[num_cols].head()

cat_but_car
df[cat_but_car].head()


##########
#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
##########
# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Bağımlı değişkenimizi binary değişkene çevirelim. (Encode da edilebilir.)
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)
# ya da
# df.loc[df["Churn"]=="Yes","Churn"] = 1
# df.loc[df["Churn"]=="No","Churn"] = 0

##########
#Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
##########
#kategorik

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col,plot=True)

#numerik
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

##########
#Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
##########
def target_summary_with_num(dataframe, target, numerical_col):       #NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

def target_summary_with_cat(dataframe, target, categorical_col):     #KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

##########
#Adım 5: Aykırı gözlem var mı inceleyiniz.
##########
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col,": ", check_outlier(df, col))

##########
#Adım 6: Eksik gözlem var mı inceleyiniz.
##########
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df.dropna(inplace= True)
# Sadece "TotalCharges" değişkeninde 11 değerin eksik olduğunu gözlemliyoruz.

# TotalCharges değişkenini float'a çevirmeden önce NaN değer görünmüyordu. Bu durumu gözlemleyelim.

index_nan = df[df.isnull().any(axis=1)].index
df2 =  pd.read_csv("Telco_Churn_Feature_Engineering/Telco-Customer-Churn.csv")
df2.iloc[index_nan]
# Bu değerler totalcharges'ta da boş görünüyor.

# TotalCharges değerleri NaN olan tüm müşterilerin tenure değerleri de 0. Ayrıca hiçbiri churn olmamış.
# Bu da bize bu müşterilerin firmanın yeni müşterisi oldukları bilgisini veriyor.
# Buna emin olmak için:
df[df["tenure"] == 0]
# Tenure değeri 0 olan tüm müşteriler aynı zamanda TotalCharges'ı NaN olan müşteriler.

###################################################
#Görev 2 : Feature Engineering
###################################################

#########
#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
#########

#eksik değerler için,
# TotalCharges değişkeninde 11 adet eksik gözlemimiz vardı.
# Toplam içinde çok az sayıda olduğu için silinebilir. 1 Aylık ödemeleri yazılabilir ya da hiç ödeme yapmadıkları için 0 yazılabilir. Sadece NaN olanların median'ı ile doldurulabilir.

# Biz 1 aylık ödemelerini yazalım.
df["TotalCharges"].fillna(df.iloc[index_nan]["MonthlyCharges"], inplace=True)

# Diğer alternatifler:

# df["TotalCharges"].dropna(inplace=True)

# Tüm NaN'lere 0 yazmak:
# df["TotalCharges"].fillna(0, inplace=True)

df.isnull().sum().any() #False

for col in num_cols:    #aykırı değerler için,
    print(col,": ", check_outlier(df, col))

# Aykırı değer olmadığını gözlemledik.

#########
#Adım 2: Yeni değişkenler oluşturunuz.
#########

# Tenure değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"


# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


# Şirket hizmet sektöründe yer aldığı için verdiği hizmetin kalitesinden memnuniyet durumu önemli.
# Memnuniyet durumunu tahmin edebilecek değişkenler oluşturalım.
# Öncelikle contract değişkenini rahat kullanabilmek adına sayısal değişkene çevirelim.

df.loc[(df['Contract'] == "Month-to-month" ), "NEW_CONTRACT"] = 1
df.loc[(df['Contract'] == "One year" ), "NEW_CONTRACT"] = 12
df.loc[(df['Contract'] == "Two year" ), "NEW_CONTRACT"] = 24

# Contract süresi bitmeden churn olanları aldığı hizmetten memnun kalmamış sayabiliriz.

df.loc[(df["NEW_CONTRACT"]==1) & (df["tenure"]<=2) & (df["Churn"]==1), "NEW_DISSATISFACTION1"] = 1
df.loc[(df["NEW_CONTRACT"]==12) & (df["tenure"]<=12) & (df["Churn"]==1), "NEW_DISSATISFACTION1"] = 1
df.loc[(df["NEW_CONTRACT"]==24) & (df["tenure"]<=24) & (df["Churn"]==1), "NEW_DISSATISFACTION1"] = 1
df["NEW_DISSATISFACTION1"] = df["NEW_DISSATISFACTION1"].fillna(0)
df["NEW_DISSATISFACTION1"].value_counts()


df.head()
df.shape   #(7032, 33)

#########
#Adım 3: Encoding işlemlerini gerçekleştiriniz.
#########
# Yeniden değişkenlerimizi türlerine göre ayıralım.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# NEW_TotalServices değişkeni cat_cols arasında yer almış fakat numeric bir değişken onun yerini değiştirelim.
cat_cols.remove("NEW_TotalServices")
num_cols.append("NEW_TotalServices")

# Churn bağımlı değişkenimiz olduğu için onu encode etmemize şu an için gerek yok.
cat_cols.remove("Churn")

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape  # (7032, 49)

#########
#Adım 4: Numerik değişkenler için standartlaştırma yapınız.
#########
num_cols
scaler = RobustScaler() # Medyanı çıkar iqr'a böl.
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

#####################################################
#Görev 3 : Modelleme
#####################################################
#########
#Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
#########
y = df["Churn"]
X = df.drop(["customerID","Churn"], axis=1)

######################
# Modeling using CART
######################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.7750292431360774
cv_results['test_f1'].mean()
# 0.5836355149709086
cv_results['test_roc_auc'].mean()
#0.7170708077264422

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
# n_estimator --> birbirinden bağımsız fit edilecek ağaç sayısı
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.836316759343075
cv_results['test_f1'].mean()
# 0.6518258572647945
cv_results['test_roc_auc'].mean()
#0.8731385454673125

################################################
# GBM (Gradient Boosting Machines)
################################################

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8393041902861228
cv_results['test_f1'].mean()
#0.6550368796926145
cv_results['test_roc_auc'].mean()
#0.8874904238053809

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.8235209847935693
cv_results['test_f1'].mean()
#0.6341751725704504
cv_results['test_roc_auc'].mean()
# 0.8744297695255483

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8333327267341406
cv_results['test_f1'].mean()
#0.6528774464502372
cv_results['test_roc_auc'].mean()
# 0.8809887757102463

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.8366031051812669
cv_results['test_f1'].mean()
#0.6531385002396479
cv_results['test_roc_auc'].mean()
# 0.8845695842803778

#####################
# Seçtiğim 4 Model
####################
# GBM
# CatBoostLGBM
# LGBM
# Random Forests

#########
#Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.
#########

################################################
# GBM (Gradient Boosting Machines)
################################################
gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()
gbm_params = {"learning_rate": [0.01, 0.1],  #ne kadar küçük ise train uzamaktadır ama daha başarılı tahmin
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}    #bütün gözlemler mi bellirli bir kısmı mı

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.8421480284009742
cv_results['test_f1'].mean()
#0.6650541911805472
cv_results['test_roc_auc'].mean()
#0.8870245031780188

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8393028759878721
cv_results['test_f1'].mean()
# 0.6586167180063056
cv_results['test_roc_auc'].mean()
# 0.8866375796069115

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8397306295185321
cv_results['test_f1'].mean()
# 0.6425385796740282
cv_results['test_roc_auc'].mean()
#0.8873655765816103

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
# n_estimator --> birbirinden bağımsız fit edilecek ağaç sayısı
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],          #derinlik
             "max_features": [3, 5, 7, "auto"],  #değişken sayısı
             "min_samples_split": [2, 5, 8, 15, 20],   #gözlem sayısı dallanmaya bağlı
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.8390182497090392
cv_results['test_f1'].mean()
#0.6413200805014622
cv_results['test_roc_auc'].mean()
#0.8799546335269675

##########
# Adım 3:  Modele en çok etki eden değişkenleri gösteriniz ve önem sırasına göre kendi belirlediğiniz kriterlerde değişken seçimi yapıp seçtiğiniz.
# değişkenler ile modeli tekrar çalıştırıp bir önceki model skoru arasındaki farkı gözlemleyiniz.
################################################
# GBM
################################################

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.8421480284009742
cv_results['test_f1'].mean()
#0.6650541911805472
cv_results['test_roc_auc'].mean()
#0.8870245031780188

#Feature Importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final , X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)













