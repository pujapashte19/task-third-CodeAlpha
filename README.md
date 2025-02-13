## Automated Essay Scoring

#### Improve upon essay scoring algorithms to improve student learning outcomes.

### Overview
This repository contains Jupyter notebooks that demonstrate the process of building and evaluating machine learning models for automated essay scoring. 
The goal is to create reliable and efficient models that can provide timely feedback to students and support educators, especially in underserved communities.


### Approaches
#### LightGBM + TF-IDF
This approach involves using LightGBM (Light Gradient Boosting Machine) with TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction to classify essay scores.

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient, with the capability to handle large-scale data with high performance and speed. In this project, LightGBM is used to build a model that can accurately predict essay scores based on features extracted from the text data.

TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. It is a numerical representation of text that transforms the raw text into features that can be used by machine learning algorithms. By applying TF-IDF, the textual data from essays is converted into a format that LightGBM can process, capturing the relevance and significance of each term within the essays.

Combining LightGBM with TF-IDF allows for an effective classification model that can process and analyze the text data to predict scores accurately. This method leverages the strengths of LightGBM in handling large datasets and the ability of TF-IDF to highlight important textual features, resulting in a robust automated essay scoring system.

#### RAPIDS SVR
RAPIDS SVR (Support Vector Regression) is utilized for regression tasks on large datasets, leveraging GPU acceleration for faster computation. RAPIDS is an open-source suite of software libraries and APIs built on CUDA, which enables execution on NVIDIA GPUs.

SVR is a type of Support Vector Machine (SVM) used for regression challenges. It seeks to find a function that deviates from the actual observed values by a value no greater than a specified margin and at the same time is as flat as possible. SVR is particularly effective in cases where the relationship between data points is complex and non-linear.

Using RAPIDS SVR in this project allows for efficient handling and processing of large-scale essay datasets, enabling faster training and prediction times due to the parallel computing capabilities of GPUs. This makes it feasible to work with extensive data, ensuring that the regression model can provide precise scoring predictions in a timely manner.
