# Fundamentals_tensorflow

### This repository provide the fundamental of Tensorflow starting from creating tensor's to making your own tensorflow models.😍

## *Part 1:- [Tensor Tutorial](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Complete_tenors_tutorial_notebook.ipynb)* 
*This is the most important section for everyone because it focuses on learning the fundamentals and understanding what operation are done in the ML model behind the scenes! So, before we begin constructing neural network models, we must first learn about the fundamentals of Tensors and  Numpy.*

**The structure of notebook is as follows:-**
1. [Creating our first tensors with TensorFlow](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=JEbuaH7GO_CK)
2. [Creating Random Tensor](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=kRG5ILQOO_CM&line=3&uniqifier=1)
3. [Play around with Order of Tensor](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=IRmPd07DO_CM)
4. [Checking the property of any tensor](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=Pe131hd_O_CO)
5. [Indexing Tensor's](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=zBf8R1RjO_CQ)
6. [Matrix/Tensor Multiplication (IMP)](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=4_TnnHSZO_CR)
7. [Datatype and Aggreation of tensor's](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=tQfpZycFO_CT&line=1&uniqifier=1)
8. [Positional Maxima and minima](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=hnMyNR_RO_CU)
9. [Edit dimension of tensor (Squeezing) and more..](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=Kmpt3-jgO_CU&line=1&uniqifier=1)
10. [Using tensorflow with Numpy](https://colab.research.google.com/drive/1wzhqe8kvVO4IHiLjop4jbcBFSZGrPQuH#scrollTo=IvGVCTMYda5o)

## *Part2:-[Neural Network Regression](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=k6oqVFrjWPK5)*
*Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type.This notebook includes the basic neural network models as we are only doing regression starting with creating a custom dataset and eventually working on Insurance dataset avaiable on kaggle.*

**The structure of notebook is as follows:-**
1. [Introduction of Neural Network](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=xGc1U1ZWGPZW)
2. [Check Input & Output Shape of any Dataset](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=d3gi4hqsJ5sv)
3. [Steps of Modeling in Tensorflow](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=k6oqVFrjWPK5)
4. [Tips to Improve our Model](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=rQUkOwz2pVLQ)
5. [Increase efficience Model](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=u_rPjZ-9uwzI)
6. [Evaluation & Visualization of Model](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=uPnIKv9SuecO)
7. [Create a Ploting function](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=BedKNZD52-am)
8. [Tracking and Saving any Model](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=AvdNGTc_Fpyw)
9. [Creating a Insurance Model](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=vZjYOdwRc4zf)
10. [Preprocesssing with Normalizatio](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=Q7PC_69WblJ5)

## *Part3:-[Neural Network Classification](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=k6oqVFrjWPK5)*
*Neural network Classification is another supervised learning method,here we classify the given value based on the dataset on which the model is created also learn about getting optimal learning rate of any optimizer,various activation layers for different dataSet, Visualizing the Classification using PlotDecisionBoundary Function,etc*

**Types of _Classification Problems_**

* `Email`-'Spam' or 'Not Spam' a perfect example of **Binary classification** present in [**Part4**]

* `ImageClassification` - like 'Cat' , 'Dog' and 'Person' etc is called **Multiclass Classification** present in [**Part5**], but used tensorflow fashion dataset rather than animal classification.

* `TagsPrediction` - in a wikipedia page "what tags should this page include?" is called **MultiLabel Classification**. present in [**Part6**]

**The structure of notebook is as follows:-**
1. [Introduction of Neural Network Classification](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=ViWNe0ci3HxP&line=7&uniqifier=1)
2. [Creating a Dataset of 2 circles using pandas](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=d3gi4hqsJ5sv)
3. [Binary Classification Model using Tensorflow](https://colab.research.google.com/drive/1cqSMkzp0-A_FQM3Pb0f8jtaUbYyJaWo9#scrollTo=k6oqVFrjWPK5)
4. [Visualize models Loss v/s Accuracy and Evaluate](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=yRqrJxvFUEJm&line=2&uniqifier=1)
5. [Create a Plot_Decision_Boundary Fuction](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=A1DpWVt-cT3z&line=4&uniqifier=1)
6. [Non-Linearity](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=gMvqGI0FxV2_)
7. [Different Tf Layers Activation Function](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=_nTh8Pi1AWai&line=2&uniqifier=1)
8. [Visualizing the test and Train Results ](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=vXzIlQgyrIs4&line=7&uniqifier=1)
9. [Finding the Best Learning Rate](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=qR0HU5z8zgNq)
10. [Prettify Our Confusion Matrix](https://colab.research.google.com/github/pravincoder/Fundamentals_tensorflow/blob/main/Part3_Tensorflow_Neural_Network_Classification.ipynb#scrollTo=94tkeO1hbSyH)
