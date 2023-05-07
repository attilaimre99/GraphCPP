# GraphCPP: A state-of-the-art graph convolutional neural network for cell-penetrating peptides

## Abstract
**TODO**

## Installation
We recommend installing GraphCPP with [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html). Mamba is a fast, robust, and cross-platform package manager. Mambaforge is a Python-based CLI conceived as a drop-in replacement for conda, offering higher speed and more reliable environment solutions.

```
mamba env create -f environment.yml
mamba activate graphcpp
```
You can deactive the environment later with `mamba deactivate`

The raw dataset is provided in the `dataset/raw` folder, it also contains the processed files zipped. If you want to skip the featurization process you can just unzip the provided zip file.

The repository has been tested on CUDA 11.7.

## Repository structure
The following folder&file structure describes the modules used in this project.

```bash
├── main.py # Quick helper file to reproduce our results.
├── hyperparameter.py # Perform bayesian hyperparameter search using arm-mango.
├── dashboard.py # Single-page streamlit server for easy prediction locally.
├── cv.py # k-fold cross validation of the selected model.
├── config.py # Configuration file. This is where we specify the best architecture.
├── graphcpp # GraphCPP module.
│   ├── act.py # Activation layers.
│   ├── dataset.py # Dataset loader and featurizer functions.
│   ├── generalconv.py # Modified version of the general convolutional layers from Design Space for Graph Neural Networks https://arxiv.org/abs/2011.08843.
│   ├── lightning.py # Lightning modules.
│   ├── model.py # Model architecture.
│   ├── pooling.py # Global graph pooling layers.
│   └── utils.py # Misc utils.
├── assets # Misc static assets.
├── model # Saved model file.
├── README.md # Readme file.
└── .gitignore # gitignore file with general python ignores.
```

## Prediction
GraphCPP utilizies the [streamlit](https://streamlit.io/) python package for locally hosting a web-server. You can start the prediction dashboard with the following command in the root directory of the project:
```
streamlit run dashboard.py
```
---

## Hyperparameter optimization
We have utilized [arm-mango](https://github.com/ARM-software/mango) to search the vast hyperparameter space of our model. In previous iterations of the model we have determined that the GraphSAGE convolutional layer performs best for our data. To perform hyperparameter optimization run the following script:
```
python run hyperparameter.py
```

To track and visualize the results we have used [mlflow](https://mlflow.org/docs/latest/index.html); MLflow is an open source platform for managing the end-to-end machine learning lifecycle. To see the results open mlflow in the root directory (*in another shell*):
```
mlflow ui
```
---

## k-fold Cross-validation
To cross-validate the best model determined by hyperparameter optimization run the following script:
```
python run cv.py
```
It defaults to 10-fold cross-validation.

---

## Citation
Please cite our paper if you have found our work helpful:
```
TODO
```