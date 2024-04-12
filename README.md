# Predicting failures in Scania trucks using advanced machine learning techniques (SMOTE, Random Forest, L1 penalized Logistic Regression) 
#### *Focusing on addressing data challenges and improving model accuracy through strategic adjustments and advanced modeling techniques
<img width="423" alt="Screen Shot 2024-04-12 at 3 39 46 PM" src="https://github.com/niharikaabhange/APS-Failure-in-Trucks-SMOTE-Random-Forest-L1-Logistic-Regression-/assets/73836890/ac4170e2-7476-4e8d-9264-b3cc4e8acad0">

#### Dataset:
The dataset for this project is derived from https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

The dataset includes a training set and a test set designed for predicting failures in Scania trucks. The training set comprises 60,000 entries with 1,000 of them belonging to the positive class, spread across 171 columns including a class column.

#### 1) Data Preparation
-> Researched techniques for handling missing values due to significant missing data in the dataset, and applied one of these techniques to manage missing values without discarding data.

-> Calculated the coefficient of variation (CV) for each of the 170 features to assess the variability of each feature relative to its mean.
Plotted a correlation matrix to visualize relationships between features and identify potential multicollinearity.

-> Selected and visualized the top features with the highest CV using scatter plots and box plots to evaluate their significance and distribution visually.

-> Determined the class distribution within the dataset to confirm if the dataset is imbalanced, which it likely is given the initial data description.

#### 2) Initial Random Forest Model
-> Trained a random forest classifier without compensating for the class imbalance initially to establish a baseline performance.

-> Evaluated the model using a confusion matrix, ROC curve, AUC, and misclassification rates for both training and test datasets. Additionally, calculated the Out of Bag error estimate to compare against the test error.

#### 3) Adjusting for Class Imbalance
-> Researched methods to handle class imbalance specific to random forests and adjusted the random forest model accordingly.

-> Re-trained the random forest model incorporating strategies for class imbalance and compared the results with the initial model to assess improvements.

#### 4) XGBoost and Model Trees
-> Explored using XGBoost to implement a model tree where L1-penalized logistic regression was used at each decision node, enhancing the generalization by incorporating all input dimensions.

-> Determined the regularization term (α) via cross-validation and trained the model tree on the APS dataset without compensating for class imbalance initially.

-> Reported on the performance metrics including the confusion matrix, ROC, and AUC, and compared the training and test errors.

#### 5) Incorporating SMOTE
-> Applied SMOTE to address the class imbalance before modeling to synthesize new examples for the minority class.

-> Re-trained the XGBoost model with L1-penalized logistic regression at each node using the SMOTE-enhanced data.

-> Carefully applied cross-validation to avoid incorrect training/test data leakage and compared the results of the SMOTE-enhanced model against the uncompensated model to highlight differences and improvements.
