import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('ccd.csv')

plt.matshow(df.corr())
plt.colorbar()
plt.show()

df.drop(['total_night_minutes','total_day_minutes','total_eve_minutes','total_intl_minutes'],1,inplace=True)

for col in ['international_plan','voice_mail_plan']:
	for i in range(len(df)):
		if df[col].iloc[i]==" yes":
			df[col].iloc[i]=1
		else:
			df[col].iloc[i]=0

df['number_vmail_messages']=df['number_vmail_messages']/np.float(np.max(df['number_vmail_messages']))
df['account_length']=df['account_length']/np.float(np.max(df['account_length']))


column = ['total_day_calls',
		'total_eve_calls',
		'total_night_calls',
		'total_intl_calls',
		'total_day_charge',
		'total_eve_charge',
		'total_night_charge',
		'total_intl_charge']

for col in column:
	mean= np.mean(df[col])
	std = np.mean(df[col])
	df[col]=(df[col]-mean)/std

col='churn'
for i in range(len(df)):
	if df[col].iloc[i]==" True":
		df[col].iloc[i]=1
	else:
		df[col].iloc[i]=0

df.to_csv('newccd.csv',index=False)
