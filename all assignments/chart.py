import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load built-in dataset
tips = sns.load_dataset("tips")
print(tips.head())


# Load built-in dataset
tips = sns.load_dataset("tips")
print(tips.head())

sns.barplot(data=tips, x="day", y="total_bill", ci=95)  # default
sns.barplot(data=tips, x="day", y="total_bill", ci="sd")  # standard deviation
sns.barplot(data=tips, x="day", y="total_bill", ci=None)  # no error bars



# sns.set_style("whitegrid")  # or try "white" or "ticks"
# sns.barplot(data=tips, x="day",ci=None, y="total_bill")

# plt.show()

# sns.barplot(data=tips, x="day", y="total_bill", color='skyblue', edgecolor='pink', linewidth=1.5)
# plt.grid(False)
# plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()



sns.lineplot(data=tips, x="total_bill", y="tip")
plt.show()


sns.barplot(data=tips, x="day", y="total_bill")
plt.show()


sns.histplot(data=tips, x="total_bill", bins=20, kde=True)
plt.show()


sns.boxplot(data=tips, x="day", y="total_bill")
plt.show()


sns.violinplot(data=tips, x="day", y="total_bill")
plt.show()


correlation = tips.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.show()


