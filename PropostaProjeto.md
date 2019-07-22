# Project Proposal

## Background and Motivation

>Discuss your motivations and reasons for choosing this project, especially any background or research interests that may have influenced your decision.

Diariamente, diversos eventos ocorrem nas vias das grandes cidades, provocando distúribios no trânsito e alterações nas rotas do transporte particular. Com o progresso da tecnologia Mobile, milhares de pessoas se utilizam de aplicativos durante seu trajeto para guiar sua rota, produzindo uma enorme quantidade de dados, os quais alguns são inputados pelos próprios usuários. Dentre esses aplicativos, um dos mais populares é o Waze, no qual seus usuários podem reportar diferentes tipos de alertas, entre eles, os de acidentes nas vias. 

Atráves dos dados de aplicativos podemos melhor compreender os padrões de mobilidade das cidades em diferentes escalas de agregação, comparar as semelhanças e diferenças entre cidades e desenvolver modelos para explicar alguns desses padrões.

## Project Objectives 
>What are the scientific and inferential goals for this project? What would you like to learn and accomplish? List the benefits.

### Objetivos de aprendizado
* Fazer estatística preditiva com dados climáticos e de trânsito.
* Fazer uma estimação não-paramétrica da densidade de um processo pontual em redes lineares.
* Fazer modelagem com dados geo-espaciais.

### Objetivos sociais
* Fazer análise exploratória dos dados
* Criar visualizações com dados de geolocalização
* Identificar padrões que possam trazer medidas preventivas ou alimentar sistemas de alertas das prefeituras.

## Must-Have Features
>These are features or calculations without which you would consider your project to be a failure.
* Capacidade de prever o quanto que eventos climáticos afetam o trânsito de uma cidade.
* Capacidade de correlacionar eventos climáticos com incidentes como enchentes/alagamentos.
* 

## Optional Features
>Those features or calculations which you consider would be nice to have, but not critical.
* Criar visualizações com dados geo-espaciais.

## What Data?
>From where and how are you collecting your data?
* Waze
* Dark Sky API: https://darksky.net/dev/docs#time-machine-request 
* OpenStreetMap: https://www.openstreetmap.org/

### Waze
A base de dados que temos disponível do Waze  armazena dados de todas capitais brasileiras, outras quatro cidades latino-americanas e o Estado da Califórnia desde outubro/novembro de 2018 em 3 tabelas, sendo duas que deverão ser consultadas.

#### Tabela Alerts
* pubMillis
* location
* type
* subtype
* street
* city
* jamUuid
* nThumbsUp

### Dark Sky
>The Dark Sky API allows you to look up the weather anywhere on the globe, returning (where available):
>* Current weather conditions.
>* Minute-by-minute forecasts out to one hour.
>* Hour-by-hour and day-by-day forecasts out to seven days.
>* Hour-by-hour and day-by-day observations going back decades.

A API possui diversas fontes de dados, tal como isd	(The USA NOAA’s Integrated Surface Database).

#### Tabela Weather Conditions
* Apparent (feels-like) temperature
* Cloud cover
* Humidity
* Liquid precipitation rate
* Precipitation type
* Temperature

### OpenStreetMaps (OSM)
OpenStreetMap é um projeto de mapeamento colaborativo para criar um mapa livre e editável do mundo, inspirado por sites como a Wikipedia.
 
## Design Overview
>List the statistical and computational methods you plan to use.
* Linear models
* kernel density estimation
  
## Verification
>How will you verify your project's results? In other words, how do you know that your project does well?
* Comparar resultados entre capitais.
* Separar a base em treino, validação (cross-validation) e teste.

## Visualization & Presentation
>How will you visualize and communicate your results?
* Usando pacotes de visualização do python e talvez programas como Tableau para tanto gerar gráficos de estatísticas, quanto para gerar mapas.
* Gráficos de precipitação e desvios da média do trânsito.
* Gerar gráficos por cidades para poder comparar.
* Plotar gráficos (scatterplots, por exemplo) para relacionar precipitação com incidentes.
  
## Schedule
>Make sure that you plan your work so that you can avoid a big rush right before the final project deadline, and delegate different modules and responsibilities among your team members. Write this in terms of weekly deadlines.
* 5/8 - Obtenção dos dados das bases
* 12/8 - Análise exploratória pronta
* 19/8 - Visualizações prontas
* 26/8 - 1o Protótipo do site
* 2/9 - Primeiros resultados de modelagem e predições
* 6/9 - Modelagem e predições prontas. Site pronto.