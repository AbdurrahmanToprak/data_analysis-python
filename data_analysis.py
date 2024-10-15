from traceback import print_exc
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import quantile

m = np.random.randint(1,30, size=(5,3))
df1 = pd.DataFrame(m, columns=["a", "b", "c"])
df2 = df1 + 99
print(pd.concat([df1, df2] , ignore_index=True)) #birbirine bağla ignore index ile indexi devam ettir sıfırlama
pd.set_option('display.max_columns', None)
df = pd.read_csv('datasets/readCSVExample.csv')
print(df.head())

df2 = sns.load_dataset('titanic')
print(df2)
print(df2.describe().T)
print(df2.isnull().values.any())
print(df2["sex"].value_counts())
print(df2[0:13])
df2 = df2.drop(1, axis=0)
print(df2.head())
df2.reset_index()
print(df2.head())
print(df2.loc[0:3, "age"])
print(df2.loc[(df2["age"] > 50) &  (df2["sex"]== "male" ), ["age", "class"]].head())
print(df2.loc[df2["age"] > 50, ["age", "class"]].head())
print(df2.groupby("sex").agg({"age" : ["mean" , "sum"]}))
print(df2.pivot_table("survived" , "sex" , "embarked"))
print(df2.pivot_table("survived" , "sex" , "embarked" , aggfunc="std"))
df2["new_age"] = pd.cut(df2["age"] , [0,10,18,25,40,90])
pd.set_option('display.width', 500)
print(df2.pivot_table("survived" , "sex" , ["new_age" , "class"]))
print(df2[["age"]].apply(lambda x:x/10).head())
df2["sex"].value_counts().plot(kind='bar')
plt.show()
plt.hist(df2["age"])
plt.show()
plt.boxplot(df2["fare"])
plt.show()

y = np.array([2,5,8,2,26])
plt.plot(y, marker='o')
plt.show()
y = np.array([2,5,8,2,26])
plt.plot(y, linestyle='dashed', marker='o', color='red')
plt.title("Ana başlık")
plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.grid(True)
plt.show()

y = np.array([2,5,8,2,26])
x = np.array([5,2,9,15,3])
plt.subplot(1,2,1)
plt.title("1")
plt.plot(x,y)
y = np.array([3,5,2,2,25])
x = np.array([5,2,9,15,3])
plt.subplot(1,2,2)
plt.title("2")
plt.plot(x,y)
plt.show()

sns.countplot(x=df2["sex"] , data=df2)
plt.show()
sns.boxplot(x=df2["fare"])
plt.show()

def check_df(dataframe, head=5):
    print("######## shape ##########")
    print(dataframe.shape)
    print("######## types ##########")
    print(dataframe.dtypes)
    print("######## head ##########")
    print(dataframe.head(head))
    print("######## tail ##########")
    print(dataframe.tail(head))
    print("######## NA ##########")
    print(dataframe.isnull().sum())
    print("######## quantiles ##########")
    print(dataframe.describe([0,0.25,0.5,0.75,0.99]).T)

check_df(df2)

cat_cols = [col for col in df2.columns if str(df2[col].dtypes) in ["object", "category" , "bool"]]
print(cat_cols)
num_but_cat = [col for col in df2.columns if df2[col].nunique() < 10 and df2[col].dtypes in ["int64", "float64"]]
print(num_but_cat)
cat_but_car = [col for col in df2.columns if df2[col].nunique() > 20 and str(df2[col].dtypes) in ["object", "category"]]
print(cat_but_car)
cat_cols = cat_cols + num_but_cat
print(cat_cols)
cat_cols = [col for col in cat_cols if col not in cat_but_car]
print(cat_cols)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df2, "sex")

for col in cat_cols:
    cat_summary(df2, col,plot=True)

num_cols = [col for col in df2.columns if df2[col].dtypes in ["int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]
print(num_cols)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles= [0, 0.10, 0.2, 0.3, 0.5, 0.75, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
num_summary(df2, "age")

for col in num_cols:
    num_summary(df2, col, plot=True)

def grab_col_names(dataframe , cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
    cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        kategorik değişken listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    --------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols un içerisinde
    Return olan 3 liste toplam değişken sayısına eşittir

    """
    cat_cols = [col for col in df2.columns if str(df2[col].dtypes) in ["object", "category", "bool"]]
    num_but_cat = [col for col in df2.columns if df2[col].nunique() < 10 and df2[col].dtypes in ["int64", "float64"]]
    print(num_but_cat)
    cat_but_car = [col for col in df2.columns if
                   df2[col].nunique() > 20 and str(df2[col].dtypes) in ["object", "category"]]
    print(cat_but_car)
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    print(cat_cols)

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols , num_cols, cat_but_car

cat_cols , num_cols, cat_but_car = grab_col_names(df2)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df2, "survived" , "sex")

for col in cat_cols:
    target_summary_with_cat(df2, "survived", col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))

for col in num_cols:
    target_summary_with_num(df2, "survived", col)
df2 = df2.iloc[:1:-1]

corr = df2[num_cols].corr() #korelasyon
print(corr)

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


cor_matrix = df2[num_cols].corr().abs()  # Mutlak değer
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.09)]
df2[num_cols].drop(drop_list, axis=1, inplace=True)
print(drop_list)

print(upper_triangle_matrix)
