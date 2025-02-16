import pandas as pd

# Script zum Mischen mehrerer csv-Daten-Dateien
#
# initial 20250216

# CSV-Dateien einlesen
df1 = pd.read_csv('daten-m.csv')
df2 = pd.read_csv('daten-b.csv')
df3 = pd.read_csv('daten-s.csv')

df1.set_index('JD', inplace=True)
df2.set_index('JD', inplace=True)
df3.set_index('JD', inplace=True)

def b(wert):
    return wert.str.replace(" €", "").str.replace(",",".").astype(float)

# Erstelle Dummydaten durch Mixen und bewerten von verschiedenen einzelnen Dateien
df_combined= 3.0*df1[['BETRAG']].apply(b) + 3.0*df2[['BETRAG']].apply(b) + 0.3*df3[['BETRAG']].apply(b)

def B(s):
    return f"{s:.2f}".replace(",", "X").replace(".", ",").replace("X", ".")+" €"

df_combined['BETRAG'] = df_combined['BETRAG'].apply(B)
df_combined['DATE'] = df1['DATE']
df_combined['JD'] = df_combined.index
df_combined['DUMMY'] = ""

df_combined = df_combined[['DATE', 'JD', 'BETRAG', 'DUMMY']]

df_combined.to_csv('daten-dummy.csv', index=False)
