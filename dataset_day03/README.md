What This Project Does

The program performs the following steps:

Loads the customer churn dataset.
Checks basic data information, including shape, data types, missing values, duplicates, and summary statistics.
Cleans the data by removing duplicates, missing values, and unreasonable values.
Creates new features such as age groups and tenure groups.
Analyzes customer churn based on age, tenure, monthly charges, and total charges.
Generates simple visualizations.
Saves the cleaned dataset.
Main Analysis

The project includes:

Top 10 oldest customers
Top 10 customers with the highest monthly charges
Top 10 customers with the highest total charges
Average monthly charges by churn status
Age and tenure summary statistics
Customer churn summary table
Generated Outputs

After running the program, the following files will be created:

output/cleaned_data.csv
figures/churn_distribution.png
figures/age_vs_totalcharges.png
How to Run

Install the required libraries:

pip install pandas matplotlib

Run the Python file:

python data03.py
Technologies Used
Python
pandas
matplotlib