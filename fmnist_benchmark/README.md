# Fashion MNIST benchmark

## Data

Information on data is avaiable [here](https://github.com/zalandoresearch/fashion-mnist)

## Benchmark

A benchmark is already available [here](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/). This benchmarl is complementary since it provides insights also on image features and not only on ML classifiers. 

## Features

- raw : raw pixel values
- daisy : DAISY features computed by the skimage function
- hog
- rocket
- unsup_km
- local
- haar
    
## Models

- rf
- gbt
- hist_gbt
- ext
- pca_nn

## Benchmark data

WIP : Current experiments were run on a subset of the data. The train test split is done with sklearn utility function with train_size=2000, test_size=2000 and random_state=13 if somebody wants to replicate or add entries to the benchmark.

|   test_accuracy |   train_accuracy | model    | features   |
|----------------:|-----------------:|:---------|:-----------|
|          0.8435 |           1      | hist_gbt | daisy      |
|          0.83   |           1      | hist_gbt | raw        |
|          0.829  |           1      | rf       | daisy      |
|          0.8225 |           1      | rf       | raw        |
|          0.811  |           1      | ext      | daisy      |
|          0.805  |           1      | hist_gbt | unsup_km   |
|          0.8025 |           1      | hist_gbt | hog        |
|          0.7895 |           1      | ext      | raw        |
|          0.7875 |           1      | rf       | hog        |
|          0.7865 |           1      | hist_gbt | rocket     |
|          0.775  |           1      | rf       | unsup_km   |
|          0.772  |           1      | ext      | hog        |
|          0.7655 |           1      | ext      | unsup_km   |
|          0.7575 |           0.843  | pca_nn   | daisy      |
|          0.7555 |           1      | rf       | rocket     |
|          0.731  |           0.818  | pca_nn   | hog        |
|          0.731  |           0.825  | pca_nn   | raw        |
|          0.6975 |           0.7955 | pca_nn   | unsup_km   |
|          0.6565 |           1      | ext      | rocket     |
|          0.649  |           0.793  | gbt      | daisy      |
|          0.6005 |           0.725  | gbt      | unsup_km   |
|          0.5635 |           0.7005 | gbt      | rocket     |
|          0.5335 |           0.6395 | gbt      | raw        |
|          0.529  |           0.7015 | pca_nn   | rocket     |
|          0.506  |           0.6775 | gbt      | hog        |
