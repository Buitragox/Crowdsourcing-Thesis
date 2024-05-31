# Deep learning with multiple annotators for breast cancer detection

This repository contains the code, notebooks and dataset used for the models trained in the thesis
"Modelo para la detección de cáncer de seno en imágenes histológicas a
partir de aprendizaje profundo con múltiples anotadores".

Made by Jhoan Buitrago and Juan González, with the help of our advisor Julián Gil González.

## Dataset

The original dataset comes from the ["Breast Cancer Semantic Segmentation (BCSS) dataset"](https://github.com/PathologyDataScience/BCSS).

The preprocessed dataset comes from ["Learning from crowds in digital pathology using scalable variational Gaussian processes"](https://github.com/wizmik12/crowdsourcing-digital-pathology-GPs) and the data can be found at this [link](https://drive.google.com/drive/folders/1yWT1aaQLiZAkAomtAdFlqlVWnRkhNrCu).

The final data used for the training of the models of this work can be found in the `/data` directory.

- `/data/pkl` contains the pickle files for majority voting and crowdsourced labels. These files have the annotations and labels.

    - Pickle files can be read using [pandas.read_pickle](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html).
    
    - `utils.py` has functions for loading the labels of gold standard, majority voting and multiple annotators.

- `/data/TrainTestNpyInt.zip` contains npy files of the preprocessed dataset after doing feature extraction with a VGG16.
