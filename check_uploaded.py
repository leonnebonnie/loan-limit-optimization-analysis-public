import pandas as pd
import os
p1 = 'loan_limit_increases (1).xlsx'
if not os.path.exists(p1):
    print('File not found:', p1)
else:
    df = pd.read_excel(p1)
    print('Loaded:', p1)
    print('Shape:', df.shape)
    if 'Customer ID' not in df.columns:
        first_row = df.iloc[0].astype(str).str.lower()
        if first_row.str.contains('customer id').any():
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            print('Adjusted header from first row')
    print('Columns:', df.columns.tolist())
    if 'Days Since Last Loan' in df.columns:
        eligible = (pd.to_numeric(df['Days Since Last Loan'], errors='coerce') >= 60).sum()
        print('Eligible (>=60 days):', int(eligible))
    else:
        print('No Days Since Last Loan column')
