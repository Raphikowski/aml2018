*Project for Advanced Machine Learning 2018 (by Lennard Kiehl & Raphael Sayer)*
# Facial Expression Recognition 2013

### Data sources
- original challenge [(kaggle)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) [(paper)](https://arxiv.org/abs/1307.0414)
- dataset: FER-2013 [(kaggle download)](https://www.kaggle.com/c/3364/download-all)
- possible better annotations: FER+ [(github)](https://github.com/Microsoft/FERPlus) [(paper)](https://arxiv.org/abs/1608.01041)

### Data description
- 35887 grayscale images (48x48x1) of faces (preprocessed)
- orignal labels include 7 classes/emotions (one per image): (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
- updated labels (FER+): 10 crowd-sourced labels per image
- split: train - 28709,  public test/eval - 3589,  test- 3589

### Performance on original labels
| Model  | Top-1 Accuracy (%) |
| ----| --- |
| "null" model  | 60  |
| ensembel of "null" models | 65.5 |
| original winner | 71.162  |
| approximate performance of vanilla CNN (found through browsing) | ~64-68 |

### Possible Architectures
- AlexNet (2012) [(paper)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- VGG (2014) [(paper)](https://arxiv.org/abs/1608.01041)
- ResNet (2015) [(paper)](https://arxiv.org/abs/1512.03385)
- Highway Networks (2015) [(paper)](https://arxiv.org/abs/1505.00387)
- DenseNet (2016) [(paper)](https://arxiv.org/abs/1608.06993)
- SqueezeNet (2016) [(paper)](https://arxiv.org/abs/1602.07360)
- ConvNet-AIG (2017) [(paper)](https://arxiv.org/abs/1711.11503)
- MobileNetV2 (2018) [(paper)](https://arxiv.org/abs/1801.04381)

