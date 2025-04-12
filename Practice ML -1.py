import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data
x = pd.read_csv("D:\\ML Practice\\medical-charges.csv")

# Group by region and sex, then count
gender_counts = x.groupby(['region', 'sex']).size().unstack()

# Plotting
gender_counts.plot(kind='bar', stacked=False)
plt.title('Count of Sex in Each Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.legend(title='Sex')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

fig = plt.hist(medical_df, x = 'age' , marginal = 'box')
p