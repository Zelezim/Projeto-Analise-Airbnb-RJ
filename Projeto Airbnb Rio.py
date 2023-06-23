#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Importar Bibliotecas e Bases de Dados

# In[15]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[16]:


meses = {'jan':1, 'fev':2, 'mar':3, 'abr':4, 'mai':5, 'jun':6, 'jul':7, 'ago':8, 'set':9, 'out':10, 'nov':11, 'dez':12}
caminho_bases= pathlib.Path('Copy of dataset\dataset')

base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)
    
#dataframe = dataframe.append(df2)    
#abril2018_df = pd.read_csv(r'Copy of dataset\dataset\abril2018.csv')
display(base_airbnb)


# alterando o tamanho da exibicao dos dados
#pd.set_option('display.height', 50)
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 50)


# - Verificando quantas colunas temos no arquivo e quanto isso pode impactar na velociadade de edicao/elituira do dados
# - Varias colunas nao sao necessarias:
# 
# 1. Id, links, informacoes irrelkavantes para o modelo
# 2. Colunas repetidas ou muito parecidas
# 3. Colunas preenchidas com texto livre
# 4. Colunas em que todos os valores ou quase todos os valores sao iguais
# 
# 

# In[17]:


print(base_airbnb[['bed_type']].value_counts())


# In[18]:


print(list(base_airbnb.columns))
base_airbnb.head(500).to_csv('primeiros registros',sep = ';')


# ### Consolidar Base de Dados

# In[19]:


colunas = ['host_response_rate','host_is_superhost','host_listings_count','host_total_listings_count','host_verifications','host_has_profile_pic','host_identity_verified','latitude','longitude','is_location_exact','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','square_feet','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','availability_30','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes'
]
base_airbnb = base_airbnb.loc[:, colunas]


# In[20]:


display(base_airbnb)


# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir

# ### Tratar Valores Faltando
# - Visualizando so dados percebemos que existe uma grande dispareidade em dados faltantes as colunas com mais de 100.000 valores Nan foram excluidas da analise.

# In[21]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 100000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
        


# In[22]:


base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())


# ### Verificar Tipos de Dados em cada coluna

# In[23]:


print(base_airbnb.dtypes)
print('-'*50)
print(base_airbnb.iloc[0])


# - Vamos ter que mudar o tipo de variavel da coluna price e extra people que estao sendo reconhecidas como objetc

# In[24]:


#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$','')
base_airbnb['price'] = base_airbnb['price'].str.replace(',','')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32)

#extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32)

print(base_airbnb.dtypes)


# ### Análise Exploratória e Tratar Outliers

# In[25]:


plt.figure(figsize=(15,10))
sns.heatmap(base_airbnb.corr(),annot=True,cmap='Greens')


# ### Definicao de funcoes para a analise de outliers
# 
# Vamos definir algumas funcoes para ajudar na analise de outliers

# In[26]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
  # returnar o limite_inferior e o limite superior
    return q1 -1.5*amplitude, q3 +1.5*amplitude

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
   #df = df.loc[linhas, colunas]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


# In[27]:


print(limites(base_airbnb['price']))
base_airbnb['price'].describe()


# In[28]:


def diagrama_caixa(coluna):
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15,5))
    sns.distplot(coluna,hist=True)
    
def grafico_barra(coluna):
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())  
    ax.set_xlim(limites(coluna))


# In[29]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imoveis comuns, acreito que os valores acima do limite superiror serao apenas de apartamentos de altissimo luxo o que nao o objeito princiapla da analise. Por isso vamos excluir esses outliers

# In[30]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print('{} linhas removidas'.format(linhas_removidas))


# In[31]:


histograma(base_airbnb['price'])


# ### extra people

# In[32]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[33]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print('{} linhas removidas'.format(linhas_removidas))


# ### host_listings_count

# In[34]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# Podemos excluir os outliers, pois o objetivo do nosso projetos nao contempla o publico  de imobiliarias ou profissionais diereiconados para a area de locacao de imoveis

# In[35]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print('{} host_listings_count'.format(linhas_removidas))


# ### accommodates 

# In[36]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# In[37]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print('{} accommodates'.format(linhas_removidas))


# ### bathrooms

# In[38]:


diagrama_caixa(base_airbnb['bathrooms'])
grafico_barra(base_airbnb['bathrooms'])


# In[39]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print('{} accommodates'.format(linhas_removidas))


# ### bedrooms 

# In[40]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# In[41]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print('{} accommodates'.format(linhas_removidas))


# ### beds

# In[42]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# In[43]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print('{} accommodates'.format(linhas_removidas))


# ### guests_included 

# In[44]:


diagrama_caixa(base_airbnb['guests_included'])
grafico_barra(base_airbnb['guests_included'])


# Vamos remover essa feature da analise. Parece que os usuarios do airbnb nao preenchem corretamente o campo de guest included, o que pode fazer o nosso modelo considerar um feature que acabar por nao ser essencial para o preco, nos levando a concluir que sera melhor excluir a coluna da analise.

# In[45]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape


# ### minimum_nights 

# In[46]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])
print(limites(base_airbnb['minimum_nights']))


# In[47]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print('{} accommodates'.format(linhas_removidas))


# ### maximum_nights 

# - A coluan desmonstra que quase todos os hosts não preenchem esse campo de maximum nights, então ele não parece que vai ser um fator relevante, por isso sera melhor exclui-lo da analise.
# 

# In[48]:


diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])


# In[49]:


base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape


# ### number_of_reviews 

# A coluna de numeros de revies sera exlcuida pelo seguintes motivos:
# 
#     1. Se excluirmos os outliers, vamos excluir as pessoas que tem a maior quantidade de reviews (o que normalmente são os hosts que têm mais aluguel). Isso pode impactar muito negativamente o nosso modelo
#     2. Pensando no nosso objetivo, se eu tenho um imóvel parado e quero colocar meu imóvel lá, é claro que eu não tenho review nenhuma. Então talvez tirar essa característica da análise pode na verdade acabar melhorando nosso modelo.

# In[50]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# In[51]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape


# ### Tratamento de dados da colunas de texto

# ### property_type  

# In[52]:


print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15,5))
grafico = sns.countplot(x='property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# - Aqui a nossa ação não é "excluir outliers", mas sim agrupar valores que são muito pequenos.
# - Todos os tipos de propriedade que têm menos de 2.000 propriedades na base de dados, eu vou agrupar em um grupo chamado "outros". Com o objetivo de simplificar e ajustar nosso modelo.

# In[53]:


tabela_tipo_casa = base_airbnb['property_type'].value_counts()
coluna_agrupar = []

for tipo in tabela_tipo_casa.index:
    if tabela_tipo_casa[tipo] <2000:
        coluna_agrupar.append(tipo)
print(coluna_agrupar)

for tipo in coluna_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'
    


# ### room_type

# In[54]:


plt.figure(figsize=(15,5))
grafico= sns.countplot(x='room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# ### bed_type  
# 

# In[55]:


plt.figure(figsize=(15,5))
grafico= sns.countplot(x='bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# ### cancellation_policy 

# In[56]:


plt.figure(figsize=(15,5))
grafico= sns.countplot(x='cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# In[57]:


#Agrupando categoraias de cancelation policy
tabela_tipo_casa = base_airbnb['cancellation_policy'].value_counts()
coluna_agrupar = []

for tipo in tabela_tipo_casa.index:
    if tabela_tipo_casa[tipo] < 10000:
        coluna_agrupar.append(tipo)
print(coluna_agrupar)

for tipo in coluna_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'scrict'


# ### amenities

# Como temos uma diversidade muito grande de amenities, onde por vezes, a mesmas amenities poodem ser escritas de forma diferente, vamos avaliar a quantidade de amenities como o parametro para o nosso modelo.

# In[58]:


print(base_airbnb['amenities'].iloc[0].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))
base_airbnb['numero_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)


# In[59]:


base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape


# ### numero_amenities

# diagrama_caixa(base_airbnb['numero_amenities'])
# grafico_barra(base_airbnb['numero_amenities'])

# - Então, essa virou uma coluna de valor numérico e, como todas as outras colunas de valores numéricos, foi excluido os outliers como nos mesmos modelos anteriores.

# In[60]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'numero_amenities')
print('{} numero_amenities'.format(linhas_removidas))


# ### Visualizacao de mapa das propriedades

# - Para fins de visualizacao sera criado um mapa que exibe uma fracao da nossa base de dados aleatório (5.000 propriedades) para verificar como as propriedades estão distribuídas pela cidade e também identificar os locais de maior preco.

# In[61]:


amostra = base_airbnb.sample(n=5000)
centro_mapa = {'lat':df['latitude'].mean(), 'lon':df.longitude.mean()}
mapa = px.density_mapbox(amostra, lat=amostra.latitude, lon=amostra.longitude, z=amostra.price, radius=2.5,
                        zoom=10,center=centro_mapa,
                        mapbox_style='stamen_terrain')
mapa.show()


# ### Encoding

# Precisamos ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true e false, etc.)
# 
# - Feature de valores Ttrue ou False, vamos substituir por 1 e False por 0.
# - Features de Categoria (features em que os valores da coluna sao textos) vamos utilizar o metodo de encoding de variaveis dummies

# In[62]:


print(base_airbnb.iloc[0])


# In[63]:


base_airbnb = base_airbnb.drop('host_total_listings_count', axis=1)
base_airbnb = base_airbnb.drop('host_verifications', axis=1)
base_airbnb = base_airbnb.drop('host_identity_verified', axis=1)
base_airbnb = base_airbnb.drop('is_location_exact', axis=1)
base_airbnb = base_airbnb.drop('host_has_profile_pic', axis=1)


# In[64]:


colunas_tf = ['host_is_superhost','instant_bookable','is_business_travel_ready' ]
base_airbnb_encoding = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_encoding.loc[base_airbnb_encoding[coluna]=='t', coluna] = 1
    base_airbnb_encoding.loc[base_airbnb_encoding[coluna]=='f', coluna] = 0
print(base_airbnb_encoding.iloc[0])    


# In[65]:


colunas_categoria = ['property_type','room_type','bed_type','cancellation_policy']
base_airbnb_encoding = pd.get_dummies(base_airbnb_encoding,columns=colunas_categoria)
display(base_airbnb_encoding.head())


# ### Modelo de Previsão

# - Metricas de avaliacao

# In[66]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\R2:{r2}\RSME{RSME}'


# - Escolha dos modelos a serem testados
# 	1. Radom Forest
# 	2. Linear Regression
#     3. Extra Tree

# In[67]:


modelos = {'RandomForest': RandomForestRegressor(),
          'LinearRegresion': LinearRegression(),
           'ExtraTree': ExtraTreesRegressor(),
          }

y = base_airbnb_encoding['price']
x = base_airbnb_encoding.drop('price', axis=1)


# - Separar os daods em treino e teste + Treino do Modelo

# In[68]:


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #Treinar
    modelo.fit(x_train, y_train)
    #Testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# ### Análise do Melhor Modelo

# In[69]:


for nome_modelo, modelo in modelos.items():
    #Testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# - Modelo Escolhido como melhor modelo: ExtraTressRegressor.
# 	<br /> Esse foi o modelo que apresentou o maior valor de R2 e ao mesmo tempo o menor valor do RSME. Como nao tivemos uma grande diferenca de velocidadede treino e de previsao desse modelo com o modelo de RandomForest (que teve os ressultados proximos de R2 E RSME, vamos escolher o modelo  ExtraTrees).
#     
#    <br /> O modelo de reressao linear nao obteve um resultado satisfatorio com valores de R2 e RSME nao satisfatorios.
#       <br />
# - Resultados das Metricas de avalicao no Modelo Vencedor: <br />
# Modelo ExtraTree: <br />
# R2:0.9691 <br />
# RSME46.593 <br />

# ### Ajustes e Melhorias no Melhor Modelo

# In[70]:


print(modelos['RandomForest'].feature_importances_)
print(x_train.columns)

importancia_features = pd.DataFrame(modelos['RandomForest'].feature_importances_,x_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)

plt.figure(figsize=(15,5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])  
ax.tick_params(axis='x', rotation=90)


# ### Ajuste Finais no Modelo
# 
# - is_buniness_travel_ready nao parece apresentar muito impacto no nosso modelo. Entao para chegar em um modelo mais simples  vamos excluir essa feature e testar o modelo sem ela.

# ## Deploy do Projeto

# In[72]:


x['price'] = y
x.to_csv('dados_deploy.csv')


# In[73]:


import joblib
joblib.dump(modelos['ExtraTree'], 'modelo.joblib')

