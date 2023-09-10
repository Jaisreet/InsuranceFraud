# FraudDetection

## Requirements
Most of the calculations and analyses in this project are done with tools that are included with Anaconda, namely:
- Pandas
- NumPy
- SciPy
- Scikit-Learn
- Matplotlib

However, we explored other algorithms that would require installing a library called 'lightgbm'.
This library can be installed using pip, with the command below.

    pip install lightgbm

## Files
The project consists of 3 main code files:
1. preparing_data.py
This file contains functions that are needed to process data before using the dataset to train the model.
There is no need to run this file, it is only a container for the functions.

2. train_model.py
The main process of the project is saved in this file. Reading the data, processing it, and training the models
are done using codes in this file. All analysis, comparisons, and model evaluations are contained here.

3. data_visualization.py
This file contains some code to visualize the data as part of elementary data analysis. 

There is also a data file provided, called 'insurance_fraud.csv'

## Running the code
The whole logic is implemented in the train_model.py file, so to run it, use the following command:

python3 train_model.py

Make sure that the file 'insurance_fraud.csv' is in the same folder.

Running this code will print the results that are included in the report. 
The first part is the result of the chi-squared test.
The part that follows prints the test score and the classification report for the model that we trained, both using oversampled and undersampled data.
The last part prints the scores of a model that does not use AgeOfPolicyHolder, which significance is tested.