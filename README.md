# MVCNN

# How to freeze layers [pytorch link](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor)
```
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.last_layer.in_features
model.last_layer = nn.Linear(num_ftrs, num_classes)

model = model.to(device)
```
For more complex situations (nn.Sequence for example)
```
model_conv.classifier._modules["2"] = nn.Linear(num_ftrs, num_classes)
```

# NEW FOR TRAINING
- Training the stages is done seperately
- Training can be resumed

# Changes made to the train_mvcnn script
- Batch Size and number of epochs has to be set explicitly via arguments.<br>
    Otherwise they will default to 4 and 1 respectively (-bs & -num_epochs)
- DataLoader workers can be set too. Default: 4 (-num_workers)
- Stage for training has to be specified (-stage)
- Specify which model (run folder) should be used for stage 2 <br>
    ("... -svcnn_name convnext_tiny_2" to use stage 1 of /runs/convnext_tiny_2/stage_1)
- When continuing, specify wandb ID (e.g. "... -resume_id vyrf9ok7")
