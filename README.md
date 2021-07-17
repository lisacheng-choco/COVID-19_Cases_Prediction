# COVID-19 Cases Prediction
## Overview
This is my assignment on [Machine Learning 2021 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html). The project is to predict the percentage of new tested positive cases in the 3rd day based on the past 3 days in a specific state in U.S. with **deep neural networks(DNN)**. 

The objectives are:
- Solve a regression problem with deep neural networks (DNN).
- Evaluate the performance of baseline model and apply training tips for further improvement.

### What I did in the project
1. I found baseline model is overfitting. 
2. I overcame overfitting by reducing the number of features, with RFECV to select important features.
3. I tuned hyper-parameters, including
    - the number of hidden layers
    - the number of layer units

The table below shows the MSE of my model and the baseline model, and it indicates that the MSE difference between training/validation sets get much more closer.
|     | My model | Baseline model |
| --- | --- | --- |
| MSE of training set | 0.73 | 0.47 |
| MSE of validation set | 0.76 | 0.75 |

### Install
This project requires Python 3.8 and the following Python libraries installed:
- [Pytorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](https://jupyter.org/)

### Code
My code is provided in `main.ipynb`. You will also be required to use `dataset/covid.train.csv` and `dataset/covid.test.shuffle.csv` dataset files to complete your work. In addition, the exported models, generated when runing `main.ipynb`, will be saved in `models/` folders. The prediction result of `covid.test.shuffle.csv` is created as `pred.csv`.  
### Run
In a terminal or command window, run the following commands:
```
jupyter notebook main.ipynb
```
## Data
Source: Delphi group @ CMU, A daily survey since April 2020 via Facebook

Features are:
- States (40 features, encoded to 0ne-hot vectors)
- COVID-like illness (4 features, percentage)
- Behavior Indicators (8 features, percentage)
- Mental Health Indicators (5 features, percentage)
- Tested Positive Cases (1 feature, percentage, this is what we want to predict)

Dataset includes:
- `covid.train.csv`: 2700 samples, 94 features (40 + 18(day1) + 18(day2) + 18(day3))
- `covid.test.shuffle.csv`: 893 samples, 93 features (40 + 18(day1) + 18(day2) + 17(day3))

## Evaluation Metric
Mean Squared Error (MSE)

## Reference
Source: Heng-Jui Chang @ NTUEE (https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)
