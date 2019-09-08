# Flood and Traffic Jam analysis

This is a final project related to the class Data Science Fundamentals at [EMAp-FGV](https://emap.fgv.br), ministered by professor Jorge Poco.

## Team members 
* [Hugo Barroso Barreto](https://www.linkedin.com/in/hugobarreto1105/)
* [Marcelo B. Barata Ribeiro](https://www.linkedin.com/in/marcelo-barata-ribeiro-213b8733/ 
)

## Background, Motivation and Objectives

Daily, many different traffic problems might occur in the streets of a city. With the rise of mobile technology, thousands of people now have access to apps to help them with their routes. While they consume data, they also generate valuable information on a massive scale. One of the most popular apps related to street traffic is Waze, where people are able to report different type of incidents, such as accidents or weather conditions.

By accessing data from different sources, we are able to understand traffic patterns from different scales of aggregation, for instance, by comparing similarities between cities, which allows us to build models to explain some of those patterns.

Our aim is to develop a data science approach which informs local governments about the mostly affected streets during heavy rains. Our task will comprehend data visualizations and prediction capabilities.

We chose this problem for two main reasons : learning and purpose. For the former, we think that the challenges brought by this task greatly benefited us by giving us new skills in fields such as data gathering, visualization and mathematical modeling with geographic data. We also had to deal with a lot of decision making and synchronized teamwork. 

For the later we believe that our results might be useful to the public sector, mainly on the local level. Our visualizations and data analysis might help to guide decision makers on addressing traffic conditions and related weather hazards. Where are the worst spots in a city traffic network during rush hours? Which streets are mostly harmed by climatic conditions such as floods and heavy rains?  We are uncovering what is not evident. We are giving a valuable resource to local governments: information.

Through this project, we sought to answer some primary questions:
* How do environmental variables affect the traffic in a city?
* What are the most critical spots in a city network during floods?
* What patterns can we find on traffic data?
* How should we transmit information to potential stakeholders?

To answer these questions, we planned  the following pipeline:
* Exploratory data analysis.
* Modeling with geo-spatial data.
* Build some visualizations from geolocation data.
* Develop predictive analysis with data of weather and traffic.
* Identify patterns which might bring some preventive measures or which might feed alert * systems to local governments.

# Files List
static: Elementos estáticos da página web (CSS e imagens estáticas).
templates: Templates html usados para renderizar o website.
hw1.pdf: Esse arquivo contém o enunciado (desafio) proposto em cada uma das etapas do projeto.
hw1.sqlite: Base de dados em SQLite criada para armazenar as informações coletadas do google scholar.
database_and_graphics.ipynb: Notebook em Python com a descrição / documentação do código usado nas etapas de criação e atualização da base de dados, assim como das visualizações inseridas no website.
database_and_graphics.py: Código em Python para criação e atualização da base de dados e das visualizações (mesmo código que está documentado no arquivo .ipynb).
routes.py: script python que inicializa o website com Flask e suas funcionalidades.
scholar.ipynb: Notebook com a documentação do código utilizado na etapa de web scraping. Também explica como fazer o download do chromedriver com as funções do módulo scrape_scholar.py.
scrape_scholar.py: Script em Python para coletar as informações do perfil do autor pesquisado no Google Scholar. Também contém uma função para fazer download automático do chromedriver (driver do google chrome necessário na etapa de web scraping)

## Files map
* data: 
* docs: some documentation about the project
* notebooks

# Requirements
## Anaconda (recommended)
We chose Python 3 as our Programming Language. Our scripts were developed using Anaconda distribution and it is. Anaconda brings with it some important python packages such as pandas, numpy and sklearn. If you are new to python, it would take you much more effort to install each package individually by yourself

You need to also install other python libraries which were essential to develop our project:
> - altair - networkx - h3 - folium - descartes - shapely - seaborn

For the instalation, you need yo type the following command in your terminal:
```sh
pip install <name_of_package>
``` 
or
```sh
conda install <name_of_package>
```
Important note: if you are using windows, you should use the terminal from Anaconda CLI or make sure that Anaconda was included to PATH.

Also, to build videos from maps, we used the follwoing programs:
* [Kepler.gl](https://kepler.gl): it is a data-agnostic, high-performance web-based application for visual exploration of large-scale geolocation data sets. It is effective to build beautiful data-driven maps and we think it might provide great impact on our audience. 
* Microsoft Power Point: an office package program. We used  it to record animations from other programs such as kepler.
* Windows Fotos: to edit our videos.

# It's All About Data
## Data Sources
Our main data sources were:
* Waze
    * It is a GPS navigation software app owned by Google. It works on smartphones and tablet computers that have GPS support. Waze describes its app as a community-driven GPS navigation app.
* [Dark Sky API](https://darksky.net/dev)
    * It provides API requests to retrieve weather data anywhere in the world. Dark Sky is backed by a wide range of weather data sources, which are aggregated together to provide the most accurate forecast possible for a given location.
* [OpenStreetMap](https://www.openstreetmap.org)
    * It is a collaborative project to create a free editable map of the world. The geodata underlying the map is considered the primary output of the project.

## Featured places
We worked with three places from South America. To collect their corresponding data, we retrieved the latitude and longitude to build our requests from the APIs. After that, we had to deal with the raw data generally in JSON format, then selected the most important features and transformed it on more readable formats such as Pandas and CSV.
* Rio de Janeiro (2nd biggest city of Brazil)
* Montevideo (capital of Uruguay)
* Miraflores (a neighborhood from Lima, capital of Peru)

## Feature selection
### Waze
The most important features from waze database which we used are
* Type: user-generated alerts such as accidents, jams, weather hazards and constructions.
* Subtype: More detailed descriptions of user-generated alerts. For instance, a weather hazard might describe flood, road ice or fog.
* Street: the streets of a city.
### Dark Sky
The weather data comprises many features. We considered only wind speed and precipitation intensity. Other variables were excluded from our analysis, such as many temperature variables, other wind variables, cloud cover and humidity. The reason is that we considered the weather hazard conditions mostly correlated to traffic incidents.

## Pre-processing
We had to do some treatment to our data. In case of Rio de Janeiro, which is a spacially big city, we noticed that the data from Dark Sky changed according to neighborhoods. To address this problem, we selected three different neighborhoods (Barra, Lagoa Rodrigo de Freitas and Meier and calculated the average value of features such as wind speed and precipitation intensity.

# Data Exploration, Predictions and Final Comments
To check for all visualizations and predicions developed in this project, we suggest that you go to the corresponding website where we placed all relevant products. 

Who should benefit the most from this project?

We envision local governments as our main stakeholders. They should be the most interested entity in consuming this type of data. The common drawbacks for them is that generally, worldwide, public sector don't have the skills to access this data . Even when it is accessible, they don't have the capabilities to interpret it and restructure into insightful visualizations so that decision makers can sketch better alert systems and prevention. Also, by understanding the most affected spots in a city network, they can also design more focused public policies. Public money is a scarce resource, so it should be used wisely and in the most efficient way, benefiting the most of the population.

That's where Data Scientist strive! Data without insights is like a raw mineral buried underground. It has no value until someone digs in and refines it. It is hard work, but in the end we are able to offer a good product, something valuable to someone.

Was this project interesting to you? Do you have any suggestions? Let us know so that we can improve it!