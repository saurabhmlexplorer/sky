
# SKY Personal Fitness Tracker 

Physical fitness is a crucial component of overall well-being. Many individuals struggle 
to determine the number of calories burned during exercise, which can affect their ability 
to manage weight and optimize workout routines. This project aims to develop a 
Personal Fitness Tracker that predicts calorie expenditure based on exercise data such 
as age, BMI, heart rate, duration, and body temperature using machine learning 
techniques. 


## Authors

- Saurabh Kumar Yadav [@saurabhmlexplorer](https://github.com/saurabhmlexplorer)


## Demo

Porject Link: https://sky-personal-fitness-tracker.streamlit.app/


## Hosting

**Server:** Streamlit Community Cloud Server


## Documentation


## Machine Learning Model Used: Random Forest Regressor  
### Introduction 
**Random Forest Regressor** is an ensemble learning method used for regression tasks, 
leveraging multiple decision trees to improve prediction accuracy and robustness. The 
algorithm was introduced by Leo Breiman in 2001 and is based on the concept of bagging 
(Bootstrap Aggregating), where multiple models are trained independently and combined 
to produce more reliable results.

### How Random Forest Regressor Works 
 
 
Random Forest Regressor constructs multiple decision trees during training and averages 
their predictions to reduce variance and improve accuracy. The key steps involved in the 
algorithm are: 
1. **Bootstrap Sampling:** A random subset of training data is drawn with replacement, 
meaning some data points may appear multiple times while others may be left out.

2. **Feature Selection & Randomness:** At each split in a decision tree, a random subset of features is selected instead of using all features. This introduces diversity among trees and prevents overfitting.

3. **Training Decision Trees:** Each decision tree is trained on its respective bootstrap sample and grown to its full depth or until a stopping criterion is met.

4. **Aggregation of Predictions:** In regression tasks, predictions from all trees are averaged to obtain the final output, reducing variance and enhancing generalization.

### Advantages of Random Forest Regressor 

1. **Reduction of Overfitting:** Random Forest mitigates overfitting by averaging multiple trees, leading to a more generalized model.

2. **Handling Non-Linearity:** The algorithm captures complex relationships in data, making it effective for non-linear regression problems. 

3. **Feature Importance Analysis:** Provides insights into feature significance, helping in feature selection and model interpretability.

4. **Robustness to Noise & Missing Values:** Due to multiple trees, the model is less sensitive to noisy data and missing values. 
5. **Parallel Processing:** Can be parallelized across multiple processors, improving computational efficiency.

### Limitations of Random Forest Regressor 

1. **Computational Complexity:** Training multiple decision trees can be resource intensive, especially for large datasets.

2. **Lack of Interpretability:** While individual decision trees are easy to interpret, a forest of trees makes interpretation challenging.

3. **Bias-Variance Tradeoff:** Although it reduces variance, bias may still be present if the base trees are weak learners.

### Hyperparameters in Random Forest Regressor 
 

The performance of the model can be tuned using various hyperparameters: 

- `n_estimators:` Number of decision trees in the forest. 
- `max_depth:` Maximum depth of each tree. 
- `min_samples_split:` Minimum number of samples required to split an internal node. 
- `min_samples_leaf:` Minimum number of samples required to be in a leaf node. 
- `max_features:` Number of features considered for each split. 
- `bootstrap:` Whether bootstrap sampling is used when building trees. 
These hyperparameters can be fine-tuned to optimize the model’s performance.

### Applications of Random Forest Regressor 

1. **Finance:** Used in stock price prediction, risk assessment, and fraud detection. 
2. **Healthcare:** Helps in disease prediction and medical diagnosis. 
3. **Marketing:** Predicts customer spending behavior and churn rate. 
4. **Energy Sector:** Forecasts electricity consumption and renewable energy generation. 
5. **Engineering:** Used in predictive maintenance and failure analysis.


## Screenshots
**Welcome Screen**
![Welcome Screen](https://github.com/saurabhmlexplorer/sky/blob/main/images/Welcome%20Screen.png)

**User Input Screen**
![User Input Screen](https://github.com/saurabhmlexplorer/sky/blob/main/images/User%20Input%20Screen.png)

**Prediction Screen**
![Prediction Screen](https://github.com/saurabhmlexplorer/sky/blob/main/images/Prediction%20Screen.png)

**Analysis Screen**
![Analysis Screen](https://github.com/saurabhmlexplorer/sky/blob/main/images/Analysis%20Screen.png)

**Personalized recommendation Screen**
![Personalized recommendation Screen](https://github.com/saurabhmlexplorer/sky/blob/main/images/Personalized%20Recommendation%20Screen.png)

