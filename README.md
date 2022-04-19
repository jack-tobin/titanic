# titanic
Titanic Passenger Survival Modelling

This was a quick project using some machine learning techniques to predict passenger survival based on characteristics such as their gender, age, ticket class, etc.

I applied a number of common ML models to the data and ultimately decided to move forward with a logistic regression as this offered consistently high recall scores and is one of the more simple ML models out there.

In testing I was able to predict survival with ~90% specificity (correct identification of 'died') and ~75% sensitivity (correct identification of 'survived') using a lostigic regression. Below is the output of my model selection process:

```python
knn; Sensitivity: 58.7%; Specificity: 90.2%
tree; Sensitivity: 79.8%; Specificity: 79.9%
forest; Sensitivity: 77.9%; Specificity: 88.4%
gnb; Sensitivity: 77.9%; Specificity: 85.4%
logit; Sensitivity: 75.0%; Specificity: 91.5%
lda; Sensitivity: 74.0%; Specificity: 91.5%
```

In applying the fitted logistic regression model to the unlabelled testing data I achieved a 0.77033 accuracy score.
