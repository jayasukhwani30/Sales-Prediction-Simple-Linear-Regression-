# Sales Prediction from Advertising Data

This project explores the relationship between advertising budgets for TV, radio, and newspaper media and sales outcomes. Using a dataset of advertising budgets and corresponding sales figures, we build a predictive model to forecast sales based on advertising spend. The project employs linear regression, a foundational method in statistical learning.

## Project Overview

The goal is to understand how advertising spending impacts sales and to predict future sales based on the advertising budget. We perform exploratory data analysis (EDA) to visualize the relationships between each advertising medium and sales, and then use linear regression to build a predictive model.

## Getting Started

### Prerequisites

This project requires Python and the following Python libraries installed:

- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

You can install these libraries using pip:

```sh
pip install pandas seaborn matplotlib scikit-learn
```

### Dataset
The dataset contains advertising budgets for TV, radio, and newspaper media, along with the sales figures. Each row represents the spending on each media and the sales for a single market.

## Analysis Steps

### Exploratory Data Analysis (EDA): 
Visualize the dataset to understand the relationship between advertising budgets and sales.

### Data Preprocessing: 
Prepare the data for modeling by splitting it into features (X) and target (y) variables.

### Model Building: 
Use linear regression to build a model that predicts sales based on advertising spend.

### Model Evaluation: 
Evaluate the model's performance using mean squared error (MSE) and the coefficient of determination (R^2).

### Results
The EDA provides insights into the correlation between advertising mediums and sales.
The linear regression model allows us to predict sales based on advertising spending accurately.
Model evaluation metrics help in understanding the model's accuracy and predictive power.
How to Run
Ensure the dataset file advertising.csv is downloaded and the file_path variable in the script points to its location on your machine.

Execute the Python script to perform the analysis and view the results.

### Conclusion
This project highlights the power of linear regression in predicting outcomes based on various inputs. Through EDA, data preprocessing, model building, and evaluation, we gain valuable insights into how advertising spending affects sales.
