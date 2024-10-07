import pandas as pd

# Load the Excel file
file_path = './Organ_Donation_and_Transplantation_Data.xlsx'
outcome_measures_df = pd.read_excel(file_path, sheet_name='OTC Outcome Measures')

# Check the first few rows of the outcome measures
print(outcome_measures_df.head())
