*Project for Advanced Machine Learning 2018 (by Lennard Kiehl & Raphael Sayer)*
# Facial Expression Recognition 2013

### Data sources
- original challenge [(kaggle)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- dataset: FER-2013 [(kaggle download)](https://www.kaggle.com/c/3364/download-all)
- possible better annotations: FER+ [(github)](https://github.com/Microsoft/FERPlus) [(paper)](https://arxiv.org/pdf/1608.01041.pdf)

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
