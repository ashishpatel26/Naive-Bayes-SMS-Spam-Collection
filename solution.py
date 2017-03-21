import pandas as pd
'''
	Import the dataset into a pandas dataframe using the read_table method.
	Because this is a tab separated dataset, we will be using '\t' as the value for the 'sep'
	argument which specifies this format.

	Also, rename the column names by specyfiing a list ['label', 'sms_message'] to the 'names' argument of read_table()
	Print the first five values of the dataframe with the new column names.
'''
df = pd.read_table('SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])


print(df.head())

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
print(df.head());