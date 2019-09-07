# Flood and Traffic Jam analysis

This is a final project related to the class Data Science Fundamentals at EMAp-FGV. Jorge Poco.

## Team members 
* Hugo Barroso Barreto 
* Marcelo B. Barata Ribeiro. 

## Background, Motivation and Objectives

Daily, many different traffic problems might occur in the streets of a city. With the rise of mobile technoogy, thousands of people now have access to apps to help them with their itineraries. While they consume data, they also generate valuable information on a massive scale. One of the most popular apps related to street traffic is Waze, where people are able to report different type of incidents, such as accidents or weather conditions.

By accessing the data from different sources, we are able to understand traffic patterns from different scales of aggregation, for instance, by comparing similarities and differences between cities, which allows us to build models to explain some of those patterns. 

* Fazer estatística preditiva com dados climáticos e de trânsito.
* Fazer uma estimação não-paramétrica da densidade de um processo pontual em redes lineares.
* Fazer modelagem com dados geo-espaciais.
* Fazer análise exploratória dos dados
* Criar visualizações com dados de geolocalização
* Identificar padrões que possam trazer medidas preventivas ou alimentar sistemas de alertas das prefeituras.

# Files List

# It's All About Data
## Data Sources
Our main data sources were:
* Waze
* [Dark Sky API](https://darksky.net/dev)
* [OpenStreetMap](https://www.openstreetmap.org)

## Featured places
We worked with three places from South America:
* Rio de Janeiro (2nd biggest city of Brazil)
* Montevideo (capital of Uruguay)
* Miraflores (a neighborhood from Lima, capital of Peru)

## Feature selection
### Waze
* pubMillis
* location
* type
* subtype
* street
* city
* jamUuid
* nThumbsUp
### Dark Sky
The weather data comprises many features. We considered only wind speed and precipitation intensity. Other variables were excluded from our analysis, such as many temperature variables, other wind variables, cloud cover and humidity. The reason is that we considered the selected variables to were the weather hazard conditions mostly correlated to traffic incidents.

## Pre-processing
We had to do some treatment to our data. In case of Rio de Janeiro, which is a spacially big city, we noticed that the data from Dark Sky changed according to neighborhoods. To address this problem, we selected three different neighborhoods (Barra, Lagoa Rodrigo de Freitas and Meier and calculated the average value of features such as wind speed and precipitation intensity.

# Visualization
We considered some visualization principles, such as the Data-Ink ratio, a concept introduced by Edward Tufte, which establishes that a good visualization should maximize the proportion of data against the "ink" used. In other words, we should avoid noise as much as possible from the information that we want our stakeholders to absorb.

* Usando pacotes de visualização do python e talvez programas como Tableau para tanto gerar gráficos de estatísticas, quanto para gerar mapas.
* Gráficos de precipitação e desvios da média do trânsito.
* Gerar gráficos por cidades para poder comparar.
* Plotar gráficos (scatterplots, por exemplo) para relacionar precipitação com incidentes.
## Color selection
* [I want hue](http://tools.medialab.sciences-po.fr/iwanthue)
* [Color Brewer](http://colorbrewer2.org)

## Tool set
https://kepler.gl

# Results
