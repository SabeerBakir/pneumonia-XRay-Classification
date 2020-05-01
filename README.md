# pneumonia-XRay-Classification
A Deep Learning model to classify pneumonia in X-Ray images.

## Dataset
The Dataset is publicly available from Kaggle and can be downloaded from [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/tasks).

## Setup

### Anaconda environment
Import the environment using the `requirements.yaml` file using the Anaconda Navigator GUI or you can enter these commands in your shell:
```bash
$ conda env create -f requirements.yaml
```


## Folder Structure
```
┌───Code/
│   ├───checkpoints/
│   │   ├───alexnet/
│   │   ├───cnn/
│   │   ├───mlp/
│   │   └───resnext/
│   ├───figs/
│   ├───logs/
│   │   ├───alexnet/
│   │   │   └───eval_logs/
│   │   ├───cnn/
│   │   │   └───eval_logs/
│   │   ├───mlp/
│   │   │   └───eval_logs/
│   │   └───resnext/
│   │       └───eval_logs/
│   ├───models/
│   └───utils/
└───Datasets/
    └───chest_xray/
        ├───test/
        │   ├───NORMAL/
        │   └───PNEUMONIA/
        ├───train/
        │   ├───NORMAL/
        │   └───PNEUMONIA/
        └───val/
            ├───NORMAL/
            └───PNEUMONIA/

```

## How to run
 - After activating the python environment, start a jupyter notebook and open `README.ipynb`.

## Contributors
- [Kacper Twardowski](https://github.com/SinfulCitrus)
- [Zachary O'Connor](https://github.com/ZacharyOConnor)
- [Sabeer Bakir](https://github.com/SabeerBakir)