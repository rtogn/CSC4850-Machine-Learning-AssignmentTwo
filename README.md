# CSC4850-Machine-Learning-AssignmentTwo
<i>Repo for Assignment 2 for CSC 4850 Machine Learning with Dr J M Banda</i>

### Contents:
-[List of questions and contents](#list-of-homework-contents)  
-[Question Summary](#question-summary)  
-[Packages Used](#packages-used)  
-[Scikit Learn Documentation Links](#scikit-learn-documentation-referenced)  
-[About The Dataset](#about-the-dataset)  


### List of homework contents
  -Question 1: Introduction and learning about the Wine Dataset<br>
  -Questions 2-6: Perceptron Algorithm<br>
  -Questions 7-11: Decision Tree Classifier<br>
  -Questions 12-16: Logistic Regression<br>
  -Question 17: Comparison of models looking at precision, recall and accuracy

### Question Summary
Each algorithm Section of questions includes the following:
1) Setting up the data, fitting the model and calculating an accuracy score for the model on the sliced test data:
![image](https://user-images.githubusercontent.com/60898339/221708462-6f42a66e-e761-4d62-b834-3b65ab02b492.png)
2) Use of the scikit-learn Classification Report to compare performance:
![image](https://user-images.githubusercontent.com/60898339/221708741-1799fc98-fa6e-467c-8662-2a0b9fa76272.png)
4) A display of the confusion matrix comparing test set labels to the models predictions:
![image](https://user-images.githubusercontent.com/60898339/221708986-106436eb-54b4-4ab5-a11e-e33a8ec49390.png)
5) A visual display of the plotting 2 attributes colored to the resulting classification for test and prediction data:
![image](https://user-images.githubusercontent.com/60898339/221709234-1d870cf2-5933-4538-ba3e-9938604601b3.png)

Each section also includes textual Q&A components for each code section. 

### Packages Used
-[Numpy](https://numpy.org/) For array splitting and composition  
-[Pandas](https://pandas.pydata.org/) For creating data frames  
-[Sklearn](https://scikit-learn.org/stable/index.html) For learning models and performance metrics  
-[Matplotlib](https://matplotlib.org/) For all graphs and plots  

### Scikit Learn Documentation Referenced
<i>Sklearn (scikit learn) was the most prevalently used package for this assignment. The following were used along with the documentation for reference</i>  
# Models:  
-[sklearn.linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)<br>
-[sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)<br>
-[sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>
# Utility:  
-[sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)<br>
-[sklearn.metrics.classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)<br>
-[sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)<br>
-[sklearn.datasets.load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)  

### About The Dataset  
This assignment utilized  a real world dataset built into Scikit learn.[It is a copy of UCI ML Wine recognition datasets](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data). The results are a summary of the chemical analysis of wines grown from different cultivars in Genoa Italy. It has thirteen chemical attributes and three classifications of wine (labeled 0 to 2).  

:Attribute Information:  
 		- Alcohol  
 		- Malic acid  
 		- Ash  
		- Alcalinity of ash  
 		- Magnesium  
		- Total phenols  
 		- Flavanoids  
 		- Nonflavanoid phenols  
 		- Proanthocyanins  
		- Color intensity  
 		- Hue  
 		- OD280/OD315 of diluted wines  
 		- Proline  

<i>(From the dataset itself)</i>  
Original Owners: 
Forina, M. et al, PARVUS - 
An Extendible Package for Data Exploration, Classification and Correlation. 
Institute of Pharmaceutical and Food Analysis and Technologies,  
Via Brigata Salerno, 16147 Genoa, Italy.  
Citation:  
Lichman, M. (2013). UCI Machine Learning Repository  
[https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,  
School of Information and Computer Science.   

# Ciao, grazie mille!  
 Roberto E. Tognoni 27/2/2023,  
 [rtognoni1@student.gsu.edu](rtognoni1@student.gsu.edu)  
![image](https://user-images.githubusercontent.com/60898339/221713103-56f7cf7c-4421-42b7-ae8a-33f74ed032dc.png)

