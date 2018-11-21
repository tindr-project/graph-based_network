# graph-based_network
The embedding and CNN classification algorithm

## run example
In directory, type:
```
python Graph2Image_CV.py
```

## input your own files
```
python Graph2Image_CV.py --input list_train_test --output output --labels random_labels --colors example_colors
```

## input format

### --input
List of paths to your subgraphs (one per line). The format of each subgraph should be space-separated, without header, and with 3 columns (node1 node2 weight).

### --output
Just the output name.

### --labels 
Binary labels (0/1) for subgraphs, one per line (name number of lines as the --input file).

### --colors
Color (optional covariate), with 2 columns, space-separated, one node per line, without header (node_name float_value, e.g. "mynode_id 8.5").
