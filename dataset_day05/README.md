# Healthcare Dataset Analysis

This project is Day 05 of the daily data processing practice series. It uses
Python, pandas, and matplotlib to clean and analyze a healthcare dataset.

## What This Project Does

The script `data05.py` performs a complete basic data analysis workflow:

- Loads `healthcare_dataset.csv`
- Checks dataset shape, data types, duplicates, missing values, and summaries
- Removes duplicated records
- Standardizes text fields such as patient names, doctors, and hospitals
- Converts admission and discharge dates to datetime values
- Converts age, room number, and billing amount to numeric values
- Fixes negative billing amounts using median billing values by medical condition
- Creates new features for length of stay, admission month, age group, billing level, and cost per day
- Generates summary CSV files
- Creates charts for patient distribution, billing, age, admissions, and test results

## Main Analysis

The analysis focuses on:

- Overall patient count, average age, billing amount, length of stay, and cost per day
- Patient counts by medical condition
- Average billing amount by medical condition
- Admission type summary
- Insurance provider billing summary
- Monthly admission trends
- Test result distribution by medical condition

## Generated Outputs

After running the script, files are saved in the `output/` folder:

- `cleaned_healthcare_dataset.csv`
- `featured_healthcare_dataset.csv`
- `overall_summary.csv`
- `condition_summary.csv`
- `admission_summary.csv`
- `insurance_summary.csv`
- `monthly_admissions.csv`
- `test_result_summary.csv`

## Generated Figures

Charts are saved in the `figures/` folder:

- `patients_by_condition.png` - pie chart of patients by medical condition
- `patients_by_admission_type.png` - pie chart of patients by admission type
- `test_results_by_condition.png` - pie charts of test results by medical condition
- `average_billing_by_condition.png` - bar chart of average billing by medical condition
- `age_distribution.png` - histogram of patient ages
- `monthly_admissions.png` - line chart of monthly admissions

## How to Run

Install the required libraries:

```bash
pip install pandas matplotlib
```

Run the project:

```bash
python data05.py
```

## Technologies Used

- Python
- pandas
- matplotlib
- Git / GitHub
