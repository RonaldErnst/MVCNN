# Meeting 2

TO-DO List:

- A:
  - Better structure for shapeNet -> Like ModelNet implementation
    - at least give stats
      - time needed
      - space needed
- I:
  - Try Inceptionv3
  - Find a base-line vanilla representation
- R:
  - Try ConvNext
  - Branch merging

# Meeting 3
Ingo: mostly training:
- resnet18 deep
- efficientnet deep
- convnext deep for feature extraction
Ronald:
- get our architecture to work with 'original' datasets
- test on shapenet55 with lower epochs
- test different architectures outside of extraction
Askar:
- combine dataset
- get generation script working
- pitch new architectures outside of extraction

# Meeting 4
Ingo:
- Training Convnext and so on
- start latex
Ronald:
- Merge branches
- test anything else (remove batchnorms)
- Get data from wandb
Askar:
- train on selfmade ShapeNet
    - use merge-r-i-1
- Merge datasets: enough time, feasible?
- Train,Val,Test splits for ModelNet&ShapeNet