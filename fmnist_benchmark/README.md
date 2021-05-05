# Fashion MNIST benchmark

## Data

Information on data is avaiable [here](https://github.com/zalandoresearch/fashion-mnist)

## Benchmark

A benchmark is already available [here](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/). This benchmarl is complementary since it provides insights also on image features and not only on ML classifiers. 

## Features

- raw : Raw pixel values
- daisy : DAISY features computed by the skimage [function](https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.daisy)
- hog : Histogram of oriented gradients using the skimage [function](https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog) 
- rocket : Random convolutions as features [article](https://arxiv.org/pdf/1910.13051.pdf)
- unsup_km : Unsupervised feature learning based on Andrew Ng [article](http://proceedings.mlr.press/v15/coates11a/coates11a.pdf)
- local : multi scale local features using the skimage [function](https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.multiscale_basic_features)
- haar : A random subset of haar features using the skimage [functions](https://scikit-image.org/docs/dev/auto_examples/applications/plot_haar_extraction_selection_classification.html#sphx-glr-auto-examples-applications-plot-haar-extraction-selection-classification-py)
    
## Models

- rf: sklearn's RandomForestClassifier
- gbt: sklearn's GradientBoostingClassifier
- hist_gbt : : sklearn's HistGradientBoostingClassifier
- ext: sklearn's ExtraTreesClassifier
- pca_nn: nearest neighbours on PCA projections 

## Benchmark data

WIP : Current experiments were run on a subset of the data. The train test split is done with sklearn utility function with train_size=2000, test_size=2000 and random_state=13 if somebody wants to replicate or add entries to the benchmark.

| features   | model            |   test_accuracy |
|:-----------|:-----------------|----------------:|
| haar       | hist_gbt         |          0.8585 |
| local      | hist_gbt         |          0.855  |
| daisy      | hist_gbt         |          0.8435 |
| raw        | hist_gbt         |          0.83   |
| daisy      | rf               |          0.8275 |
| raw        | rf               |          0.8145 |
| haar       | rf               |          0.808  |
| local      | rf               |          0.8075 |
| daisy      | ext              |          0.8075 |
| unsup_km   | hist_gbt         |          0.805  |
| hog        | hist_gbt         |          0.8025 |
| raw        | ext              |          0.797  |
| haar       | ext              |          0.7885 |
| rocket     | hist_gbt         |          0.7865 |
| hog        | rf               |          0.786  |
| hog        | proximity_forest |          0.785  |
| daisy      | proximity_forest |          0.78   |
| local      | multiboosting    |          0.7725 |
| haar       | multiboosting    |          0.771  |
| unsup_km   | rf               |          0.7705 |
| unsup_km   | ext              |          0.769  |
| hog        | ext              |          0.7685 |
| daisy      | multiboosting    |          0.765  |
| local      | proximity_forest |          0.763  |
| raw        | proximity_forest |          0.7595 |
| local      | ext              |          0.759  |
| raw        | multiboosting    |          0.758  |
| daisy      | pca_nn           |          0.7575 |
| rocket     | rf               |          0.756  |
| unsup_km   | multiboosting    |          0.74   |
| hog        | multiboosting    |          0.736  |
| local      | pca_nn           |          0.733  |
| hog        | pca_nn           |          0.731  |
| raw        | pca_nn           |          0.731  |
| haar       | proximity_forest |          0.7265 |
| unsup_km   | proximity_forest |          0.7155 |
| haar       | pca_nn           |          0.715  |
| hog        | rotation_forest  |          0.7125 |
| rocket     | multiboosting    |          0.7025 |
| unsup_km   | pca_nn           |          0.6975 |
| haar       | rotation_forest  |          0.6805 |
| raw        | rotation_forest  |          0.679  |
| daisy      | rotation_forest  |          0.675  |
| local      | rotation_forest  |          0.6615 |
| rocket     | ext              |          0.6535 |
| unsup_km   | rotation_forest  |          0.639  |
| rocket     | rotation_forest  |          0.5955 |
| unsup_km   | gbt              |          0.565  |
| haar       | gbt              |          0.565  |
| rocket     | gbt              |          0.556  |
| rocket     | pca_nn           |          0.529  |
| daisy      | gbt              |          0.514  |
| local      | gbt              |          0.5035 |
| raw        | gbt              |          0.4965 |
| rocket     | proximity_forest |          0.4885 |
| hog        | gbt              |          0.1455 |

## Awesome resources

- timm
- fastai
- pytorch