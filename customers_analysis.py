import pandas as pd
import pybnesian as pb
import itertools
import numpy as np
from radar_chart_discrete import ComplexRadar
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sklearn
import pybnesianCPT_to_df
from operator import itemgetter
from statistics import mean
import itertools
import discrete_analysis_hellinger
import networkx as nx
import matplotlib.pyplot as plt
from discrete_representation import clusters_dag
np.seterr(divide='ignore', invalid='ignore')


#Cargamos la red y los datos
customers_red=pb.load(r"networks\best_network_customers_2.pickle")
dataframe=pd.read_csv(r'C:\Users\Victor Alejandre\Desktop\CIG\Implementación\customers.csv')
dataframe=dataframe.drop('ID',axis=1)
#Preprocesado de los datos donde convertimos en categórica la variables continuas income y age
def ni(x):
    if x<50000:
        return 'Low income'
    elif 52200<=x<130000:
        return 'Middle income'
    else:
        return 'High income'

def na(x):
    if x<26:
        return 'Junior'
    elif 26<=x<55:
        return 'Adult'
    else:
        return 'Senior citizen'


new_income=[]
new_age=[]
for i in range(dataframe.shape[0]):
    new_income.append(ni(dataframe['Income'][i]))
    new_age.append(na(dataframe['Age'][i]))
dataframe['Income']=new_income
dataframe['Age']=new_age

#Cambio del formato de las variables int a variables tipo string para poder trabajar con ellas
for var in dataframe.columns:
    dataframe[var]=dataframe[var].apply(lambda x: str(x))


#El nº de clusters se establece manualmente
n_clusters=2


for var in dataframe.columns:
    dataframe[var] = dataframe[var].astype('category')

#Generamos los clusters c1,c2,...
clusters=[]
for i in range(1,n_clusters+1):
    clusters.append('c'+f'{i}')

#Generamos el diccionario con las categorias codificadas en nº enteros para su posterior en el radar_chart
df_categories=discrete_analysis_hellinger.df_to_dict(dataframe)
#En particular para las variables categoricas de income y age como son ordinales establecemos el diccionario a mano para que se respete este orden en el display del radar chart
df_categories['Income']={'Low income':1,'Middle income':2,'High income':3}
df_categories['Age']={'Junior':1,'Adult':2,'Senior citizen':3}


#Necesitamos los posibles valores de cada variable para trabajar con los algoritmos implementados, estos deben ir en un diccionaro {variable: [categorias]}
categories={'cluster':clusters}
for var in dataframe.columns:
    categories[var]=dataframe[var].cat.categories.tolist()



#Obtenemos los representantes con las respectivas prob condicionales respecto al cluster de cada valor que toma cada variable de los representantes.
#Para el probabilistic logic sampling seleccionamos un tamaño de muestra de 100000
maps=discrete_analysis_hellinger.get_MAP(customers_red,clusters,100000)


#Generamos los representantes sin las probabilidades condicionadas
maps_values=pd.DataFrame(index=clusters)
for column in maps.columns:
    maps_values[column]=[x[0] for x in maps[column]]




#Obtenemos las importancias
importances_1={}
for cluster in clusters:
    x=maps_values.loc[[cluster]].values.tolist()[0]
    importances=dict(sorted(discrete_analysis_hellinger.importance_1(customers_red,x,categories,clusters).items(), key=lambda x:x[1],reverse=True))
    importances_1[cluster]=importances




#Construimos el Radar Chart con los representantes las importancias sin las probabilidades condicionadas
discrete_analysis_hellinger.naming_categories(maps_values,importances_1,df_categories)

#Construimos el Radar Chart con los representantes las importancias y las probabilidades condicionadas
discrete_analysis_hellinger.naming(maps,importances_1,customers_red)

#Obtenemos la representación de la red para cada cluster con los nodos coloreados según las importancias de cada variable para el representante calculado
clusters_dag(customers_red,importances_1,clusters)


