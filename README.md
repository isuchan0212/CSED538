# MLP
## Initial LR 0.1, Batch Size 128, Epochs 200
### No Scheduling
- Epoch 200, Validation Loss: 5.4698018615722654
- Epoch 200, Validation Accuracy: 0.5457
- Epoch 200, Test Accuracy: 0.5405

### StepLR
- Epoch 200, Validation Loss: 2.5333524360656736
- Epoch 200, Validation Accuracy: 0.546
- Epoch 200, Test Accuracy: 0.5357

### ExponentialLR
- Epoch 200, Validation Loss: 1.6751722684860229
- Epoch 200, Validation Accuracy: 0.3989
- Epoch 200, Test Accuracy: 0.4063

### PolynomialLR
- Epoch 200, Validation Loss: 1.4246255523681641
- Epoch 200, Validation Accuracy: 0.5084
- Epoch 200, Test Accuracy: 0.5065

### CosineAnnealing
- Epoch 200, Validation Loss: nan
- Epoch 200, Validation Accuracy: 0.0973
- Epoch 200, Test Accuracy: 0.1

### CosineAnnealingWarmup
- Epoch 200, Validation Loss: 3.9135471878051757
- Epoch 200, Validation Accuracy: 0.5388
- Epoch 200, Test Accuracy: 0.537

### AdaptiveScheduler(amsgrad)
- Epoch 200, Validation Loss: 2.306160486602783
- Epoch 200, Validation Accuracy: 0.0963
- Epoch 200, Test Accuracy: 0.1