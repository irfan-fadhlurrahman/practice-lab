
## Machine Learning Bootcamp
### [Assignment 1: Taxpayer's Political Party](https://github.com/irfan-fadhlurrahman/practice-lab/blob/main/dphi-ml-bootcamp/taxpayer-political-party/assignment-1-taxpayer.ipynb)
* The objective of this project is to predict the taxpayer political party based on their attributes. This is a multi-class classification task.
* The target variable is balance with each count around 250 and have three classes such as Democrat, Republican, and Independent.
* The dataset features have a less statistical predictive power due to each feature category per target class have same mean and similar distribution.
* The features and target are processed before modelling. 
    * All categorical variables and `AHHAge` are encoded by using Target Encoders.
    * Create a new feature, `delta_HHI_HHDL`, a difference between HHI and HHDL.
    * Label the target variable classes to integer from 0 to 2.
* At modelling stage, we observed the following model gave the best accuracy.
    * LogisticRegression: 45.6%
    * RandomForestClassifier: 43.1%
    * VotingClassifier: 43.0% 
* By evaluating the feature importances of Random Forest model, we found the following features are the most important feature to the model.
    * `AHHAge`
    * `HHI` 
    * `delta_HHI_HHDL`
    * `HHDL`
* The accuracy of model are too low due to the following reasons:
  * Insufficient number of observations. We need to gain more data by doing survey again.
  * Most of features do not have a statistical power due to their category of each classes tend to have same mean and similar distribution, even though there are no outliers present. To address this issue, we need to conduct hypothesis testing and resampling data.


### [Assignment 2: Cancer Death Rate](https://github.com/irfan-fadhlurrahman/practice-lab/blob/main/dphi-ml-bootcamp/cancer-death-rate/assignment-2-cancer.ipynb)
* The aim of this project to predict the cancer death rate based on patient attributes and global demographic data.
* The target variable, cancer death rate, is near-to-normal distributed with slightly long tail in right-side and peak around 175-190.
* Correlation between all numerical features and target variable are in the range between -0.48 to 0.47.
* Collinear variables exist in this dataset as shown on the correlation heatmap of numerical variables.
* There are 14 pairs of highly correlated variables with set threshold 0.8.
* The features and target are preprocessed before modelling phase.
    * Drop categorical variables and features that have missing values.
    * Transform all numerical variables by using log-transformer
    * Scale all numerical features by using MinMaxScaler
* The ridge regression that are used such as Ridge and Bayesian Ridge. The MSE of cross-validation result show that Bayesian Ridge have a lower MSE which is around 166.1.
* By using Bayesian Ridge model, the MSE of test set is 205.50
* RFE with LassoCV as estimator provide the result as follows.
    * The number of features really affect the MSE score. With less than 10 features result a bigger error.
    * The MSE between 10 and 27 features have a small difference (around 11).


### [Assignment 3: Travel Insurance Claim](https://github.com/irfan-fadhlurrahman/practice-lab/blob/main/dphi-ml-bootcamp/travel-insurance-claim/src/notebook.ipynb)
