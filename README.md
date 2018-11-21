# Neural network for subgraph classification
An embedding and CNN classification algorithm for subgraph classification.

# Algorithm
- Input = subgraphs, subgraph labels, node colors
- For each subgraph:

  1. Compute graph embedding using node2vec (random walks + word2vec algorithm), ndimensions = 128
  2. Reduce to a 2D dimensional space discretized into a 2D grid using generative topographic mapping (GTM), ugtm implementation
  3. For a subgraph 2D image (grid), the first channel is node density, the other channels covariates
  
- Run CNN classification algorithm, with following layers:

  1. ZeroPadding2D((3, 3))
  2. Conv2D(32, (7, 7), strides=(1, 1))
  3. BatchNormalization
  4. Relu activation
  5. MaxPooling2D((2, 2))
  6. Flatten
  7. Dense layer with sigmoid activation
 

## Run example (10-fold cross-validation)
```
python Graph2Image_CV.py
```

## Use your own files
```
python Graph2Image_CV.py --input list_train_test --output output --labels random_labels --colors example_colors
```

## Input format description

### --input
List of paths to your subgraphs (one per line). The format of each subgraph should be space-separated, without header, and with 3 columns (node1 node2 weight).

### --output
Just the output name.

### --labels 
Binary labels (0/1) for subgraphs, one per line (name number of lines as the --input file).

### --colors
Covariate, with 2 columns, space-separated, one node per line, without header (node_name float_value, e.g. "mynode_id 8.5"). There should be as many lines as nodes. At the moment, only one covariate is allowed. This will change in the next version.

### Version
1.0.0

### Requirements
- tensorflow
- keras
- ugtm
- networkx
- gensim
- numpy
