# Projeto-Analise-Airbnb-RJ

Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns

Contexto:
No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária. 
Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.

Objetivo:
Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.

As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro

Extracao de dados:
No nosso projeto foi obtido os dados descritos no link acima, onde os mesmo foram ajustados (limpeza dos dados), em seguida foi realizado uma analise exploratoria de forma dinamica e visual atraves de graficos e valores.
Apos a analise exploratoria foi feito a modelagem para a metrica no calculo do modelo de previsao. Assim foi escolhido tres modelos a serem testados (RandomForest, LinearRegression e Extra Trees). Em seguida foi verificado atraves dos resultados que o modelo Extra Trees apresetaram o melhor resultado de R2 (proporcao da variancia da variavel independente pelo modelo) e RSME (mensuracao do desvio padrao residual), depois foi apresentado a relacao de cada variavel com o valor final de alugel do imovel.

Ao Final como interacao extra foi feito um deploy do projeto em joblib para que o calculo do valor aproximado do imovel pudesse ser feito pelo usuario.
