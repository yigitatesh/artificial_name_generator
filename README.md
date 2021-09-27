# Artificial Name Generator
Artificial intelligence based English name generator <br>

## Description
You can type initial characters for your artificial names and generate names that do not exist in real life. 
The structure of generated names will be similar to real English names <br>

A random probability distribution is added to name generation phase to generate different names each time. <br>

Full code for this project can be found [here](https://github.com/yigitatesh/artificial_name_generator/blob/main/full_code/english_name_generator.ipynb).

## Data
Names dataset that consists of 5,163 first names is used ([Link](https://github.com/smashew/NameDatabases/blob/master/NamesDatabases/first%20names/us.txt) for the dataset). <br>
85% of the data is used as training data and 15% of the data is used as validation data to prevent overfitting real names. <br>

## Model
An **LSTM** model is used to learn and generate names. <br>

## Usage
Go to [app link](https://unseen-turkish-name-generator.herokuapp.com/) to open this app. <br>

1. Type initial characters for the name(s) to be generated. (You can leave it empty to not indicate any initial character)
2. Type the number of names to be generated. (You can leave it empty to generate just one name)
3. Click to "Generate" button.

After generating names, you can download these names in txt format.

## Examples

