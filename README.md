## Machine Learning Bootcamp
### [Assignment 1: Taxpayer's Political Party](https://github.com/irfan-fadhlurrahman/practice-lab/blob/main/dphi-ml-bootcamp/assignment-1/assignment-1-taxpayer.ipynb)
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


### [Assignment 2: Cancer Death Rate](https://github.com/irfan-fadhlurrahman/practice-lab/blob/main/dphi-ml-bootcamp/assignment-2/assignment-2-cancer.ipynb)
