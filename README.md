# EnergyPrediction-ASHRAE
This is a capstone project for Probability and Statistics for Data Science 2 (DS-GA 3001) at [NYU Center for Data Science](https://cds.nyu.edu/). 

## Project Intro/Objective
The purpose of this project is to model the Energy consumption of the buildings depending on their basic configuration, weather conditions and usage patterns

### Methods Used
* Inferential Statistics
* Data Visualization
* Predictive Modeling
* Counterfactual Analysis

### Technologies
* Python
* Pandas, jupyter
* Sklearn

## Motivation
The cost of cooling skyscrapers in the summer is not only high in dollars but also has a considerable environmental impact. However, significant investments are being made to improve building efficiencies to reduce costs and emissions. An accurate model of metered building energy usage( in the areas of chilled water, electric, hot water, and steam meters) will assist in this goal by giving an expected improvement in energy efficiency. Therefore, the project goal is to improve building efficiencies, reduce costs, and decrease environmental impact by developing accurate models of metered building energy usage.
 
Data Source:
* [ASHRAE - Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)

This the largest comprehensive energy dataset with over 20 million points of training data from 2,380 energy meters collected for 1,448 buildings from 16 sources.
In this project, we are working with three types of datasets:
- Buildings Data - This contains some of the building-level attributes such as nearest weather station, building_id (foreign key for building energy usage data), primary_use (indicator of the primary category of activities for the building based on EnergyStar property type definitions), square_feet (gross floor area of the building), year_built (year building was opened), and floor_count (number of floors of the building). This data is useful for attributing the energy usage to some building characteristics.
- Building Energy Usage - The building energy usage data contains four main features: building_id (foreign key for building metadata), meter (meter id code for electricity, chilled water, steam, and hot water), timestamp (when the measurement was taken), and meter_reading (energy consumption in kWh or equivalent). This data will be used to model the energy usage for buildings
- Weather Data - This contains several features such as air temperature, cloud coverage, dew temperature,hourly precipitation depth, sea level pressure, wind direction, and wind speed, collected from a meteorological station as close as possible to the building site. This data will be used to investigate the relationship between weather conditions and building energy usage, which is important for understanding energy consumption patterns and identifying potential opportunities for energy conservation.

Inferential Statistics:
* TBD

Energy Prediction:
* TBD


Findings and analysis: TBD

### Team Members

|Name     |  Github   | 
|---------|-----------------|
|[John Sutor](https://www.linkedin.com/in/johnsutor3/)|  [johnsutor](https://github.com/johnsutor)       |
|[Sharad Dargan](https://www.linkedin.com/in/sharaddargan/) |  [sharaddargan](https://github.com/sharad5)    |
