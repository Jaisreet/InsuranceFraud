import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample 

data = pd.read_csv("insurance_fraud.csv")
#print(data.head())
print(data.dtypes)
y = data['FraudFound_P']
X = data.drop('FraudFound_P', axis=1)

# deleting rows
missing_values_count = data.isnull().sum() # none

#checking for duplicates
duplicates = data.duplicated()
# Count the number of duplicate rows
num_duplicates = duplicates.sum()
# Print the number of duplicate rows
print("Number of duplicate rows:", num_duplicates)
# drop the duplicated rows:
data= data.drop_duplicates()

FraudOnly = data[data['FraudFound_P'] == 1]
#print(FraudOnly['FraudFound_P'].count())
make = FraudOnly.groupby(['Make']).size().reset_index(name='Count')
#print(make)

# deleted the columns which were not related by looking at histograms
columns_to_drop = ['PolicyNumber', 'Deductible'
                   , 'Days_Policy_Accident', 'Days_Policy_Claim'
                   , 'AgeOfPolicyHolder', 'Age', 'RepNumber', 'Month'
                   , 'WeekOfMonth', 'DayOfWeek', 'MonthClaimed'
                   , 'WeekOfMonthClaimed', 'DriverRating', 'Year', 'DayOfWeekClaimed']
data.drop(columns_to_drop, axis=1, inplace=True) #18 columns left

# Number of plots to display in each figure
plots_per_figure = 9

# Calculate the total number of figures needed
num_figures = -(-len(data.columns) // plots_per_figure)  # Ceiling division

for figure_index in range(num_figures):
    # Create a new figure for each set of plots
    plt.figure(figsize=(20, 10))
    
    # Calculate the range of columns to plot for the current figure
    start_index = figure_index * plots_per_figure
    end_index = min((figure_index + 1) * plots_per_figure, len(data.columns))
    
    for i, column in enumerate(data.columns[start_index:end_index]):
        ax = plt.subplot(3, 3, i + 1)
        col = FraudOnly.groupby([column]).size().reset_index(name='Count')
        ax.bar(col[column], col['Count'])
        ax.set_title(column)
        plt.setp(ax.get_xticklabels(), rotation=45)
        #print(column, data.groupby([column]).size())

    # Adjust spacing between subplots to make them more readable
    plt.tight_layout()
    
    # Display the current figure
    plt.savefig("Figure"+str(figure_index)+".png", format='png')

