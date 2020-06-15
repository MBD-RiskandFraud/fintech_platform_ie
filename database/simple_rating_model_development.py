import pandas as pd
import numpy as np
import glob

#... READING ALL DATA FRAMES

big_df = pd.DataFrame() 
    
for filename in glob.glob("./Rating_*.xlsx"):
#    connres.commit()
    print ("READING DATABASE: "+filename)
    df = pd.read_excel(open(filename,'rb'), sheet_name="Resultados",header = None) #Reading SABI Export without index  
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


#for i in range(len(ratings)):
#    ratings_dict[ratings.iloc[i,0]] = ratings.iloc[i,1]
#    
# _h0, _h1, _h2
# _h0: history 0, here h0 means the year 2017 (historia 0, aquí h0 significa el año 2017)
# _h1: history -1, here h1 means the year 2016 (historia -1, aquí h1 significa el año 2016)
# _h2: history -2, here h2 means the year 2015 (historia -2, aquí h2 significa el año 2015)

# Ebita Margin - Ebitda / Turn over (Ventas)
# p49100: Profit (Resultado del ejercicio)
# p40800: Amortization (Amortización) 
# p40100: Sales Turnover (Ingresos de Explotación)
# p40500: Other sales (Otros Ingresos)
df['ebitda_income'] = (df.p49100_h0+df.p40800_h0)/(df.p40100_mas_40500_h0) 


# Total Debt / Ebita 
# p31200: Short Term Debt / Deuda a corto plazo
# p32300: Long Term Debt / Deuda a largo plazo
# p49100: Profit (Resultado del ejercicio)
# p40800: Amortization (Amortización) 
df['debt_ebitda'] =(df.p31200_h0 + df.p32300_h0) /(df.p49100_h1+df.p40800_h0) 

# rraa_rrpp: Financial leveraging / apalancamiento financiero 
# p10000: Total Assets / Total activos
# p20000: Own Capital / Patrimonio neto
df['rraa_rrpp'] = (df.p10000_h0 - df.p20000_h0) /df.p20000_h0


# Log of Operating Income
df['log_operating_income'] = np.log(df.p40100_mas_40500_h0)


df['debt_equity'] =(df.p31200_h0 + df.p32300_h0) /df.p20000_h0


# Assets Turnover - Sales / Total Assets
df['assets_turnover'] = (df.p40100_mas_40500_h0) /df.p10000_h0

# Return on Assets - Net Income / Total Assets 
df['ROA'] = (df.p40100_mas_40500_h0) / df.p10000_h0

# Operating Profit Margin - Operating Profit / Turnover
df['op_margin'] = (df.p49100_h0)/ (df.p40100_mas_40500_h0) 

# Current Ratio - Current Assets / Current Liabilities
df['current_ratio'] = df.p10000_h0 / (df.p10000_h0 - df.p20000_h0)

# Gross Financial Debt / Total Assets
df['debt_assets'] = (df.p10000_h0 - df.p20000_h0) /df.p10000_h0

# Own Capital / Total Assets
df['cap_assets'] = df.p20000_h0 / df.p10000_h0

df['industry'] = 0
df.loc[df.cnae!='','industry'] = df[df.cnae!=''].apply(lambda x: int(x['cnae'][0:2]),axis = 1)

def define_act(x):
    if x.industry == 46: return 'Wholesale_trade'
    elif x.industry in [45,47,48]: return 'Retail_trade'
    elif x.industry in [41,42,43,44]: return 'Construction'
    elif x.industry in range(10,35): return 'Manufacturing'
    elif x.industry in range(64,68): return 'Financial_services'
    elif x.industry in range(49,55): return 'Transport'
    else: return 'Others'
    
df['activity'] = df.apply(define_act,axis=1)

df_mod = df[['p10000_h0', 'p20000_h0', 'p31200_h0','p32300_h0', 'p40100_mas_40500_h0',
       'p40800_h0', 'p49100_h0','target_status',
       'ebitda_income', 'debt_ebitda', 'rraa_rrpp', 'log_operating_income',
       'debt_equity', 'assets_turnover', 'ROA', 'op_margin', 'current_ratio',
       'debt_assets', 'cap_assets','activity']]

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=3,n_estimators=150)
df_clean = df_mod.replace([np.inf, -np.inf], np.nan).dropna()
#df_clean = df[['ebitda_income','debt_ebitda','rraa_rrpp','log_operating_income','target_status']].replace([np.inf, -np.inf], np.nan).dropna()
#Manufacturing
X = df_clean[df_clean.activity=='Manufacturing'].drop(['target_status','activity'],axis=1)
y = df_clean.loc[df_clean.activity=='Manufacturing','target_status']
fitted_model_manu = model.fit(X, y)

#Others
X = df_clean[df_clean.activity=='Others'].drop(['target_status','activity'],axis=1)
y = df_clean.loc[df_clean.activity=='Others','target_status']
fitted_model_other = model.fit(X, y)

#Retail trade
X = df_clean[df_clean.activity=='Retail_trade'].drop(['target_status','activity'],axis=1)
y = df_clean.loc[df_clean.activity=='Retail_trade','target_status']
fitted_model_ret = model.fit(X, y)

#Wholesale_trade
X = df_clean[df_clean.activity=='Wholesale_trade'].drop(['target_status','activity'],axis=1)
y = df_clean.loc[df_clean.activity=='Wholesale_trade','target_status']
fitted_model_whole = model.fit(X, y)


#Construction
X = df_clean[df_clean.activity=='Construction'].drop(['target_status','activity'],axis=1)
y = df_clean.loc[df_clean.activity=='Construction','target_status']
fitted_model_con = model.fit(X, y)

#Financial_services
X = df_clean[df_clean.activity=='Financial_services'].drop(['target_status','activity'],axis=1)
y = df_clean.loc[df_clean.activity=='Financial_services','target_status']
fitted_model_fin = model.fit(X, y)

#Transport
X = df_clean[df_clean.activity=='Transport'].drop(['target_status','activity'],axis=1)
y = df_clean.loc[df_clean.activity=='Transport','target_status']
fitted_model_tran = model.fit(X, y)

### Prediction on the data df_clean
#if X.activity == 'Wholesale_trade': return 
#elif X.activity == 'Retail_trade': return 
#elif X.activity == 'Construction': return 
#elif X.activity == 'Manufacturing': return 
#elif X.activity == 'Financial_services': return 
#elif X.activity == 'Transport': return 
#else: return 'Others'
#
#
#
#
#
#fitted_model = model.fit(X, y)
#y_pred = fitted_model.predict(X)
#y_pred_proba = fitted_model.predict_proba(X)[:,1]
#
#print ("ASSESSING THE MODEL...")
## CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
#from sklearn.metrics import roc_auc_score
#gini_score = 2*roc_auc_score(y, y_pred_proba)-1
#print ("GINI DEVELOPMENT=", gini_score)
#
#from sklearn.metrics import accuracy_score
#print("Accuracy: {0}".format(accuracy_score(y_pred,y)))

print ("SAVING THE PERSISTENT MODEL...")
from joblib import dump#, load
dump(fitted_model_manu, 'Rating_RandomForestClassifier_manu.joblib') 
dump(fitted_model_con, 'Rating_RandomForestClassifier_con.joblib') 
dump(fitted_model_fin, 'Rating_RandomForestClassifier_fin.joblib') 
dump(fitted_model_other, 'Rating_RandomForestClassifier_other.joblib') 
dump(fitted_model_ret, 'Rating_RandomForestClassifier_ret.joblib')
dump(fitted_model_whole, 'Rating_RandomForestClassifier_whole.joblib') 
dump(fitted_model_tran, 'Rating_RandomForestClassifier_tran.joblib') 


with open("train_features.txt", 'w') as f:
    for s in list(X.columns):
        f.write(str(s) + '\n')

# i=0
# time_in_datetime = datetime.strptime(df.fecha_cambio_estado.iloc[i], "%Y-%m-%d)
    