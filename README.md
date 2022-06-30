# MVCNN

# How to freeze layers
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
