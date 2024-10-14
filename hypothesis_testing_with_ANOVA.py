#one-way ANOVA test will enable you to determine if there is a statistically significant difference in sales among groups.

# Import libraries and packages.
#type:ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn import datasets

data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows to know about variables in the dataset
print(data.head())
#To Select Independent Variation within groups
sns.boxplot(x = "TV", y = "Sales", data = data)
plt.show()
sns.boxplot(x = "Influencer", y = "Sales", data = data)
plt.show()


data = data.dropna(axis=0)
data.isnull().sum(axis=0)

ols_formula = 'Sales ~ C(TV)'
# Create an OLS model.
OLS = ols(formula = ols_formula, data = data)
# Fit the model.
model = OLS.fit()
# Save the results summary.
model_results = model.summary()
# Display the model results.
print(model_results)



# Calculate the residuals.
residuals = model.resid
# Create a 1x2 plot figure.
fig, axes = plt.subplots(1, 2, figsize = (8,4))
# Create a histogram with the residuals.
sns.histplot(residuals, ax=axes[0])
# Set the x label of the residual plot.
axes[0].set_xlabel("Residual Value")
# Set the title of the residual plot.
axes[0].set_title("Histogram of Residuals")
# Create a QQ plot of the residuals.
sm.qqplot(residuals, line='s',ax = axes[1])
# Set the title of the QQ plot.
axes[1].set_title("Normal QQ Plot")
# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()
# Show the plot.
plt.show()


# Create a scatter plot with the fitted values from the model and the residuals.

fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)
# Set the x axis label
fig.set_xlabel("Fitted Values")
# Set the y axis label
fig.set_ylabel("Residuals")
# Set the title
fig.set_title("Fitted Values v. Residuals")
# Add a line at y = 0 to visualize the variance of residuals above and below 0.
fig.axhline(0)
# Show the plot
plt.show()

print(model_results)
# Create an one-way ANOVA table for the fit model.
print(sm.stats.anova_lm(model, typ=2))

# Perform the Tukey's HSD post hoc test.
tukey_oneway = pairwise_tukeyhsd(endog = data["Sales"], groups = data["TV"])
# Display the results
print("Tukey Summary")
print(tukey_oneway.summary())

