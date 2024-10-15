import seaborn as sns

df = sns.load_dataset("titanic")

print(df['sex'].value_counts())
print(df.nunique())
print(df['pclass'].nunique())
print(df[['pclass', 'parch']].nunique())
print(df['embarked'].dtype)
df['embarked'] = df['embarked'].astype('category')
print(df['embarked'].dtype)
print(df[df['embarked'] == "C"])
print(df[df['embarked'] != "S"])
print(df[(df['age'] < 30) & (df['sex'] == 'female')])
print(df[(df['fare'] > 500) | (df['age'] > 70)])
print(df.isnull().sum())
df = df.drop('who', axis=1)
#deck_mode = df['deck'].mode()[0]
#df['deck'].fillna(deck_mode, inplace=True)
age_median = df['age'].median()
#df['age'].fillna(age_median, inplace=True)
survived_stats = df.groupby(['pclass', 'sex'])['survived'].agg(['sum', 'count', 'mean'])
print(survived_stats)
# 30 yaş altındakilere 1, 30 ve üstüne 0 veren fonksiyon
def age_flag(age):
    return 1 if age < 30 else 0

# Titanic veri setine 'age_flag' değişkenini ekleme
df['age_flag'] = df['age'].apply(lambda x: age_flag(x))
# Tips veri setini tanımlama
tips_df = sns.load_dataset("tips")
# 'Time' değişkenine göre 'total_bill' değerlerinin toplamını, min, max ve ortalamasını bulma
time_stats = tips_df.groupby('time')['total_bill'].agg(['sum', 'min', 'max', 'mean'])
print(time_stats)
# Günlere göre 'total_bill' değerlerinin toplamını, min, max ve ortalamasını bulma
day_stats = tips_df.groupby('day')['total_bill'].agg(['sum', 'min', 'max', 'mean'])
print(day_stats)
# Lunch zamanındaki ve kadın müşterilere ait total_bill ve tip değerlerinin toplamı, min, max ve ortalaması
lunch_women_stats = tips_df.loc[(tips_df['time'] == 'Lunch') & (tips_df['sex'] == 'Female'), ['total_bill', 'tip']]
lunch_women_summary = lunch_women_stats.agg(['sum', 'min', 'max', 'mean'])
print(lunch_women_summary)
# size'ı 3'ten küçük ve total_bill'i 10'dan büyük olan siparişlerin ortalamasını bulma
average_order = tips_df.loc[(tips_df['size'] < 3) & (tips_df['total_bill'] > 10), 'total_bill'].mean()
print("Ortalama Total Bill:", average_order)
# Her bir müşterinin ödediği total_bill ve tipin toplamını veren yeni bir değişken oluşturma
tips_df['total_bill_tip_sum'] = tips_df['total_bill'] + tips_df['tip']
# total_bill_tip_sum değişkenine göre büyükten küçüğe sıralama ve ilk 30 kişiyi yeni bir DataFrame'e atama
top_30_orders = tips_df.sort_values(by='total_bill_tip_sum', ascending=False).head(30)
print(top_30_orders)

