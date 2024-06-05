# Deep learning with multiple annotators for breast cancer detection

This repository contains the code, notebooks and dataset used for the models trained in the thesis
"Modelo para la detección de cáncer de seno en imágenes histológicas a
partir de aprendizaje profundo con múltiples anotadores".

Made by Jhoan Buitrago and Juan González, with the help of our advisor Julián Gil González.

## Dataset

The original dataset comes from the ["Breast Cancer Semantic Segmentation (BCSS) dataset"](https://github.com/PathologyDataScience/BCSS).

The preprocessed dataset comes from ["Learning from crowds in digital pathology using scalable variational Gaussian processes"](https://github.com/wizmik12/crowdsourcing-digital-pathology-GPs) and the data can be found in this [google drive folder](https://drive.google.com/drive/folders/1yWT1aaQLiZAkAomtAdFlqlVWnRkhNrCu).

The final data used for the training of the models of this work can be found in this [link](https://drive.google.com/file/d/1XeVC0FOmv_V8jY31JP73yXqa4q27EWJS/view?usp=drive_link). The data is stored in a zip file which contains npy files of the preprocessed dataset after doing feature extraction with a VGG16.

- npy files can be read using [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html)

- `utils.py` has functions for loading npy files with their corresponding labels.

In `/data/pkl` you can find the pickle (.pkl) files for majority voting and crowdsourced labels. These files have the annotations and labels.

- pkl files can be read using [pandas.read_pickle](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html).
    
- `utils.py` has functions for loading the labels of gold standard, majority voting and multiple annotators.

## Repository structure

- `/notebooks` has jupyter notebooks used for the training of multiple models

    - `/notebooks/old_notebooks` has notebooks with trainings of previous models using different methodologies that were discarded for a variaty of reasons. The main reason being that they were very time consuming to train with the available hardware.

- `/data/pkl` has .pkl files with the majority voting and crowdsourced labels.

- `grid_search.py` has functions for performing grid search and saving the results of the model evaluation.

- `utils.py` contains general functions and utilities for reading and loading data.


## Final results

The results of the evaluations of each model are stored in JSON files which can be found in this [google drive folder](https://drive.google.com/drive/folders/1QPlaRgthOfBil7KBWMp3-q2B913F39_B?usp=sharing).

The files contain a list of each of the trained models with their respective hiperparameters and the evaluation reports of each of the 10 repetitions.


