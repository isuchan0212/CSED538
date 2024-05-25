# MLP
## Initial LR 0.1, Batch Size 128, Epochs 200
### No Scheduling
- Epoch 200, Train Loss: 0.00025320214645471425
- Epoch 200, Train Accuracy: 1.0
- Epoch 200, Validation Loss: 5.4698018615722654
- Epoch 200, Validation Accuracy: 0.5457
- Epoch 200, Test Accuracy: 0.5405
### StepLR
- Epoch 200, Train Loss: 0.05752259050011635
- Epoch 200, Train Accuracy: 0.995
- Epoch 200, Validation Loss: 2.5333524360656736
- Epoch 200, Validation Accuracy: 0.546
- Epoch 200, Test Accuracy: 0.5357

### ExponentialLR
- Epoch 200, Train Loss: 1.6619695875167846
- Epoch 200, Train Accuracy: 0.40395
- Epoch 200, Validation Loss: 1.6751722684860229
- Epoch 200, Validation Accuracy: 0.3989
- Epoch 200, Test Accuracy: 0.4063

### PolynomialLR
- Epoch 200, Train Loss: 1.2430114562988281
- Epoch 200, Train Accuracy: 0.563425
- Epoch 200, Validation Loss: 1.4246255523681641
- Epoch 200, Validation Accuracy: 0.5084
- Epoch 200, Test Accuracy: 0.5065

### CosineAnnealing
- Epoch 200, Train Loss: nan
- Epoch 200, Train Accuracy: 0.100675
- Epoch 200, Validation Loss: nan
- Epoch 200, Validation Accuracy: 0.0973
- Epoch 200, Test Accuracy: 0.1

### CosineAnnealingWarmup
- Epoch 200, Train Loss: 0.006620404307544231
- Epoch 200, Train Accuracy: 0.99985
- Epoch 200, Validation Loss: 3.9135471878051757
- Epoch 200, Validation Accuracy: 0.5388
- Epoch 200, Test Accuracy: 0.537


### AdaptiveScheduler(amsgrad)
- Epoch 200, Train Loss: 2.3075737674713133
- Epoch 200, Train Accuracy: 0.09775
- Epoch 200, Validation Loss: 2.306160486602783
- Epoch 200, Validation Accuracy: 0.0963
- Epoch 200, Test Accuracy: 0.1