
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.


# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?


# Soru 3: Kaç unique PRICE vardır?


# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?


# Soru 5: Hangi ülkeden kaçar tane satış olmuş?




# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?




# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?


# Soru 8: Ülkelere göre PRICE ortalamaları nedir?


# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?


# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?


#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################


#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.



#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
# agg_df.reset_index(inplace=True)



#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'




#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.



#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,


#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?


# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/persona.csv')
print(df.head())
print(df.info())
print(df.columns)
print(df["SOURCE"].nunique())
print(df["SOURCE"].value_counts())
print(df["PRICE"].nunique())
print(df["PRICE"].value_counts())
print(df["COUNTRY"].value_counts())
print(df.groupby("COUNTRY")["PRICE"].sum())
print( df.groupby("SOURCE").size())
print(df.groupby("COUNTRY")["PRICE"].mean())
print(df.groupby("SOURCE")["PRICE"].mean())
print(df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean())
agg_df = df.groupby(["COUNTRY", "SOURCE","SEX", "AGE"])["PRICE"].mean().sort_values(ascending=False).reset_index()
print(agg_df)
df["AGE_CAT"] = pd.cut(df["AGE"],
                       bins=[0, 18, 30, 45, 60, 100],  # Yaş aralıkları belirleniyor.
                       labels=["0-18", "19-30", "31-45", "46-60", "60+"],  # Aralıklara etiket veriliyor.
                       right=False)  # Sağ sınır dahil edilmez.

# COUNTRY, SOURCE ve AGE_CAT kırılımında PRICE ortalamalarını hesaplayalım ve sıralayalım.
agg_df = df.groupby(["COUNTRY", "SOURCE", "AGE_CAT"])["PRICE"].mean().sort_values(ascending=False).reset_index()

# Çıktıyı yazdıralım.
print(agg_df)
# 1. Gerekli sütunları seçip birleştirme işlemi yapıyoruz.
df["customers_level_based"] = df["COUNTRY"].str.upper() + "_" + df["SOURCE"].str.upper() + "_" + df["SEX"].str.upper() + "_" + df["AGE_CAT"].astype(str)

# 2. Aynı profillerin tekilleştirilmesi ve PRICE ortalamalarının hesaplanması
agg_df = df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()

# 3. Ortaya çıkan tabloyu PRICE’a göre azalan sırayla sıralıyoruz.
agg_df = agg_df.sort_values(by="PRICE", ascending=False)

# 4. Sonucu yazdıralım.
print(agg_df)


agg_df['SEGMENT'] = pd.qcut(agg_df["PRICE"], 4, labels = ['Segment 1', 'Segment 2', 'Segment 3', 'Segment 4'])


segment_summary = agg_df.groupby('SEGMENT')['PRICE'].agg(['mean', 'max', 'sum']).reset_index()

print(segment_summary)
# Belirtilen müşteri profilleri için segment ve gelir tahmini yap
# Örnek segment ve fiyat ortalamalarını çıkar
segment_summary = df.groupby('customers_level_based')['PRICE'].agg(['mean']).reset_index()

# Belirtilen müşteri profilleri için segment ve gelir tahmini yap
# 33 yaşında Android kullanan bir Türk kadını
turkish_android_33 = segment_summary[segment_summary['customers_level_based'].str.contains('TUR_ANDROID_FEMALE')]

# 35 yaşında iOS kullanan bir Fransız kadını
french_ios_35 = segment_summary[segment_summary['customers_level_based'].str.contains('FRA_IOS_FEMALE')]

# Sonuçları yazdır
print(f"33 yaşında Android kullanan bir Türk kadının ait olduğu segment: {turkish_android_33['customers_level_based'].values}, beklenen ortalama gelir: {turkish_android_33['mean'].values}")
print(f"35 yaşında iOS kullanan bir Fransız kadının ait olduğu segment: {french_ios_35['customers_level_based'].values}, beklenen ortalama gelir: {french_ios_35['mean'].values}")