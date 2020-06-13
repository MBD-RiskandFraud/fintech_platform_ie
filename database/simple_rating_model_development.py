import pandas as pd
import numpy as np
import glob

#... READING ALL DATA FRAMES

big_df = pd.DataFrame() 
    
for filename in glob.glob("./Rating_*.xlsx"):
#    connres.commit()
    print ("READING DATABASE: "+filename)
    df = pd.read_excel(open(filename,'rb'), sheetname="Resultados",header = None) #Reading SABI Export without index  
    df.columns = ['id', 'nombre_x', 'nif', 'nombre', 'provincia', 'calle', 'telefono', 'web', 'desc_actividad', 'cnae', 'cod_consolidacion', 'rating_grade_h2', 'rating_grade_h1', 'rating_grade_h0', 'rating_numerico_h2', 'rating_numerico_h1', 'rating_numerico_h0', 'modelo_propension_h2', 'modelo_propension_h1', 'modelo_propension_h0', 'guo_nombre', 'guo_id_bvd', 'guo_pais', 'guo_tipo', 'estado_detallado', 'fecha_cambio_estado', 'fecha_constitucion', 'p10000_h0', 'p10000_h1', 'p10000_h2', 'p20000_h0', 'p20000_h1', 'p20000_h2', 'p31200_h0', 'p31200_h1', 'p31200_h2', 'p32300_h0', 'p32300_h1', 'p32300_h2', 'p40100_mas_40500_h0', 'p40100_mas_40500_h1', 'p40100_mas_40500_h2', 'p40800_h0', 'p40800_h1', 'p40800_h2', 'p49100_h0', 'p49100_h1', 'p49100_h2']
    df['h0_anio'] = 2017     
    df = df.fillna('')
    df=df.drop(df.index[0]) #Dropping SABI variable names.
    df['nif'] = df.nif.str.upper() #CONVERTING cif INTO UPPERCASE
    for partida in ['p10000_h0', 'p10000_h1', 'p10000_h2', 'p20000_h0', 'p20000_h1', 'p20000_h2', 'p31200_h0', 'p31200_h1', 'p31200_h2', 'p32300_h0', 'p32300_h1', 'p32300_h2', 'p40100_mas_40500_h0', 'p40100_mas_40500_h1', 'p40100_mas_40500_h2', 'p40800_h0', 'p40800_h1', 'p40800_h2', 'p49100_h0', 'p49100_h1', 'p49100_h2']:
#        print (partida,"ha sido convertido en numerico")
        df[partida] = pd.to_numeric(df[partida], errors='coerce').fillna(0)- 0.005
    df['nif_normalizado'] = df['nif'].str[-8:]    
    big_df = big_df.append(df, ignore_index=True)     


df = big_df
df['target_status'] = [0 if i in ['Activa', ''] else 1 for i in df['estado_detallado']] # 0 si Activa, 1 si algo raro!

# Adding ordinal values for ratings
ratings  = pd.read_csv('ratings.csv')
ratings_dict = {}
for i in range(len(ratings)):
    ratings_dict[ratings.iloc[i,0]] = ratings.iloc[i,1]
    
# _h0, _h1, _h2
# _h0: history 0, here h0 means the year 2017 (historia 0, aquí h0 significa el año 2017)
# _h1: history -1, here h1 means the year 2016 (historia -1, aquí h1 significa el año 2016)
# _h2: history -2, here h2 means the year 2015 (historia -2, aquí h2 significa el año 2015)

# Ebita Margin - Ebitda / Turn over (Ventas)
# p49100: Profit (Resultado del ejercicio)
# p40800: Amortization (Amortización) 
# p40100: Sales Turnover (Ingresos de Explotación)
# p40500: Other sales (Otros Ingresos)
df['ebitda_income_0'] = (df.p49100_h0+df.p40800_h0)/(df.p40100_mas_40500_h0) 
df['ebitda_income_1'] = (df.p49100_h1+df.p40800_h1)/(df.p40100_mas_40500_h1)
df['ebitda_income_2'] = (df.p49100_h2+df.p40800_h2)/(df.p40100_mas_40500_h2)

# Total Debt / Ebita 
# p31200: Short Term Debt / Deuda a corto plazo
# p32300: Long Term Debt / Deuda a largo plazo
# p49100: Profit (Resultado del ejercicio)
# p40800: Amortization (Amortización) 
df['debt_ebitda_0'] =(df.p31200_h0 + df.p32300_h0) /(df.p49100_h1+df.p40800_h0) 
df['debt_ebitda_1'] =(df.p31200_h1 + df.p32300_h1) /(df.p49100_h1+df.p40800_h1)
df['debt_ebitda_2'] =(df.p31200_h2 + df.p32300_h2) /(df.p49100_h1+df.p40800_h2)

# rraa_rrpp: Financial leveraging / apalancamiento financiero 
# p10000: Total Assets / Total activos
# p20000: Own Capital / Patrimonio neto
df['rraa_rrpp_0'] = (df.p10000_h0 - df.p20000_h0) /df.p20000_h0
df['rraa_rrpp_1'] = (df.p10000_h1 - df.p20000_h1) /df.p20000_h1
df['rraa_rrpp_2'] = (df.p10000_h2 - df.p20000_h2) /df.p20000_h2

# Log of Operating Income
df['log_operating_income_0'] = np.log(df.p40100_mas_40500_h0)
df['log_operating_income_1'] = np.log(df.p40100_mas_40500_h1)
df['log_operating_income_2'] = np.log(df.p40100_mas_40500_h2)

df['profitgrowth_1'] = ((df.p49100_h0+df.p40800_h0)/(df.p49100_h1+df.p40800_h1))-1
df['profitgrowth_2'] = ((df.p49100_h1+df.p40800_h1)/(df.p49100_h2+df.p40800_h2))-1

df['debt_equity_0'] =(df.p31200_h0 + df.p32300_h0) /df.p20000_h0
df['debt_equity_1'] =(df.p31200_h1 + df.p32300_h1) /df.p20000_h1
df['debt_equity_2'] =(df.p31200_h2 + df.p32300_h2) /df.p20000_h2

# Creating a modified dataframe
# Select the features to train the model with

df_mod = df[['provincia','rating_grade_h2','rating_grade_h1', 'rating_grade_h0', 'rating_numerico_h2','rating_numerico_h1',
    'rating_numerico_h0', 'modelo_propension_h2','modelo_propension_h1', 'modelo_propension_h0',
    'p10000_h0', 'p10000_h1','p10000_h2', 'p20000_h0', 'p20000_h1', 'p20000_h2', 'p31200_h0',
    'p31200_h1', 'p31200_h2', 'p32300_h0', 'p32300_h1', 'p32300_h2','p40100_mas_40500_h0', 'p40100_mas_40500_h1',
    'p40100_mas_40500_h2','p40800_h0', 'p40800_h1', 'p40800_h2', 'p49100_h0', 'p49100_h1',
    'p49100_h2', 'h0_anio', 'nif_normalizado', 'target_status',
     'ebitda_income_0', 'ebitda_income_1', 'ebitda_income_2',
       'debt_ebitda_0', 'debt_ebitda_1', 'debt_ebitda_2', 'rraa_rrpp_0',
       'rraa_rrpp_1', 'rraa_rrpp_2', 'log_operating_income_0',
       'log_operating_income_1', 'log_operating_income_2']]

# Adding annual change ratios

df_mod['p10000_h0/1'] = df_mod['p10000_h0']/df_mod['p10000_h1']
df_mod['p10000_h1/2'] = df_mod['p10000_h1']/df_mod['p10000_h2']
df_mod['p20000_h0/1'] = df_mod['p20000_h0']/df_mod['p20000_h1']
df_mod['p20000_h1/2'] = df_mod['p20000_h1']/df_mod['p20000_h2']
df_mod['p31200_h0/1'] = df_mod['p31200_h0']/df_mod['p31200_h1']
df_mod['p31200_h1/2'] = df_mod['p31200_h1']/df_mod['p31200_h2']
df_mod['p32300_h0/1'] = df_mod['p32300_h0']/df_mod['p32300_h1']
df_mod['p32300_h1/2'] = df_mod['p32300_h1']/df_mod['p32300_h2']
df_mod['p40100_mas_40500_h0/1'] = df_mod['p40100_mas_40500_h0']/df_mod['p40100_mas_40500_h1']
df_mod['p40100_mas_40500_h1/2'] = df_mod['p40100_mas_40500_h1']/df_mod['p40100_mas_40500_h2']
df_mod['p40800_h0/1'] = df_mod['p40800_h0']/df_mod['p40800_h1']
df_mod['p40800_h1/2'] = df_mod['p40800_h1']/df_mod['p40800_h2']
df_mod['p49100_h0/1'] = df_mod['p49100_h0']/df_mod['p49100_h1']
df_mod['p49100_h1/2'] = df_mod['p49100_h1']/df_mod['p49100_h2']

# Adding ordinal values to the ratings
df_mod.rating_grade_h2 = df.rating_grade_h2.map(ratings_dict)
df_mod.rating_grade_h1 = df.rating_grade_h1.map(ratings_dict)
df_mod.rating_grade_h0 = df.rating_grade_h0.map(ratings_dict)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=3,n_estimators=1000)

#df_clean = df[['ebitda_income','debt_ebitda','rraa_rrpp','log_operating_income','target_status']].replace([np.inf, -np.inf], np.nan).dropna()
df_clean = df_mod.drop(['provincia','nif_normalizado'],axis=1).replace([np.inf, -np.inf], np.nan).dropna()
X = df_clean.drop('target_status',axis=1)
y = df_clean['target_status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 1)

fitted_model = model.fit(X_train, y_train)
y_pred = fitted_model.predict(X_test)
y_pred_proba = fitted_model.predict_proba(X_test)[:,1]

print ("ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(y_test, y_pred_proba)-1
print ("GINI DEVELOPMENT=", gini_score)

from sklearn.metrics import accuracy_score
print("Accuracy: {0}".format(accuracy_score(y_pred,y_test)))

print ("SAVING THE PERSISTENT MODEL...")
from joblib import dump#, load
dump(fitted_model, 'Rating_RandomForestClassifier.joblib') 



# i=0
# time_in_datetime = datetime.strptime(df.fecha_cambio_estado.iloc[i], "%Y-%m-%d)
    