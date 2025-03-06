# 🏅NBA Draft Prediction Model🏀
Author: _Thirada Tiamklang 14337188_

_Kaggle Compitition AT1 - 36120 Advanced Machine Learning Application - Spring 2023_

The NBA Draft is a highly anticipated annual event where basketball teams select players from American colleges and international professional leagues to join their rosters. For aspiring basketball players, getting drafted into the NBA is a significant milestone in their careers.

This project is all about leveraging machine learning to predict whether a college basketball player will be drafted into the NBA based on their statistics for the current season. The goal is to provide valuable insights and predictions for basketball enthusiasts, sports commentators, and anyone interested in the NBA draft.

## ✔️ Project Overview 
------------
Objective: Develop a predictive model that determines the likelihood of a college basketball player being drafted by an NBA team.

Metric for Model Assessment: The primary metric used to assess the performance of our prediction model is AUROC (Area Under the Receiver Operating Characteristic Curve). AUROC is a widely recognized measure in classification tasks that helps evaluate how well our model distinguishes between drafted and non-drafted players.

## 📁 Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



## 🗝️ Key Features

_Data Collection:_ We gather extensive statistics and information about college basketball players for the current season.

_Machine Learning:_ We employ various machine learning techniques and algorithms to build a robust predictive model. The model is trained on historical data to learn the patterns and factors that influence NBA draft selections.

_Evaluation:_ Our model's performance is evaluated using AUROC, which provides a comprehensive view of its classification capabilities.

## ▶️ Getting Started

If you're interested in using or contributing to this project, follow these steps:

1. __Clone the Repository:__ Begin by cloning this repository to your local machine using git clone.

2. __Pre-trained Models:__ Navigate to the ['models/'](models/) directory. You will find pre-trained models for NBA draft prediction.

3. __Predictions:__ Use the provided script (e.g., [predict.py](src/models/predict_model.py)) to make predictions on new data or customize it to suit your specific needs. This script loads the pre-trained models and is ready to predict the likelihood of college players being drafted into the NBA.

```bash
python predict.py --input new_data.csv --output predictions.csv
```
- Replace new_data.csv with your input data containing statistics of college basketball players for the current season.
- The predictions will be saved to predictions.csv (you can specify a different output file).

4. __Evaluate the Results:__ Review the predictions and assess the likelihood of college players being drafted based on the provided models.

5. __Contributions:__ If you're interested in contributing to the project or making improvements, feel free to submit issues, suggest enhancements, or contribute to the codebase.
------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
