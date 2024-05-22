# CSED538
## Initial LR 1e-3, Batch Size 128, Epochs 200
# MLP
## No Scheduling
- Epoch 200, Train Loss: 0.1896202319383621
- Epoch 200, Validation Loss: 2.6615993186950684
- Accuracy of the network on the 10000 test images: 51.91%

## StepLR
- Epoch 200, Train Loss: 1.0593642812728883
- Epoch 200, Validation Loss: 1.3280518951416016
- Accuracy of the network on the 10000 test images: 53.36%

## ExponentialLR
- Epoch 200, Train Loss: 0.2043705070734024
- Epoch 200, Validation Loss: 2.6081344913482667
- Accuracy of the network on the 10000 test images: 50.29%

## PolynomialLR
- Epoch 200, Train Loss: 0.38521708545684813
- Epoch 200, Validation Loss: 1.6887716354370117
- Accuracy of the network on the 10000 test images: 53.34%

## CosineAnnealing
- Epoch 200, Train Loss: 0.7720536209106446
- Epoch 200, Validation Loss: 1.520654432296753
- Accuracy of the network on the 10000 test images: 51.61%

## CosineAnnealingWarmup
- Epoch 200, Train Loss: 0.5327390156269074
- Epoch 200, Validation Loss: 1.762715402984619
- Accuracy of the network on the 10000 test images: 51.99%

## AdaptiveScheduler(amsgrad)
- Epoch 200, Train Loss: 0.0032111080203205346
- Epoch 200, Validation Loss: 7.006144863891602
- Accuracy of the network on the 10000 test images: 49.60%