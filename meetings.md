# Meeting 2

TO-DO List:

A:
- Better structure for shapeNet -> Like ModelNet implementation
  - at least give stats
    - time needed
    - space needed

I:
- Try Inceptionv3
- Find a base-line vanilla representation

R:
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

# Meeting 5
Ingo:
- train convnext_tiny_deep & alexnet on UnifiedDataset
- Maybe train convnext with extra conv layers
- Report: Experiments, End / Summary / Future Work

Ronald:
- Train alexnet and convnext_tiny_deep on Shapenet
- Report: Method, Related Work

Askar:
- Train Alexnet and convnext_tiny_deep on ModelNet (shaded)
- Report: Abstract, Introduction

Alterations:
- Mean instead of max
- CNN instead of pooling
- Bigger classification layer
- Cnn before pooling
- whatever you can think of

# Meetings 6
Ingo:
- Prepare presentation
- Train & test resnet with nopool and mean on unified
- Finish report parts

Ronald:
- Test convnext on shapenet
- Train VGG16 on shapenet no alterations
- If time allows do convnext with mean and no pool again for 20 epochs
- Know everything on a basic level

Askar:
- Test Modelnet networks
- Train & test resnet18 with mean and no pool on modelnet
- Know everything on a basic level
- Finish report parts