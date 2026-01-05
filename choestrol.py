import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters based on NHANES/CDC data
mean_chol = 199  # mg/dL
std_chol = 40    # mg/dL
bins = np.arange(50, 405, 5)  # Cholesterol levels from 50 to 400 in steps of 5

# Simulate normal distribution
chol_values = np.random.normal(loc=mean_chol, scale=std_chol, size=100000)
hist, bin_edges = np.histogram(chol_values, bins=bins, density=True)

# Convert density to percentage
population_perc = hist * 100 * (bin_edges[1] - bin_edges[0])
cholesterol_level = (bin_edges[:-1] + bin_edges[1:]) / 2

# Save to CSV
df = pd.DataFrame({
    'cholesterol_level': cholesterol_level,
    'population_perc': population_perc
})
df.to_csv('cholesterol_distribution.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(cholesterol_level, population_perc, label='Cholesterol Distribution', color='blue')
plt.xlabel('Total Cholesterol Level (mg/dL)')
plt.ylabel('Population Percentage (%)')
plt.title('Distribution of Total Cholesterol Levels\nUS Males Age 40–60 (NHANES 2021–2022)')

# Add arrow at 184 mg/dL
arrow_x = 184
arrow_y = np.interp(arrow_x, cholesterol_level, population_perc)
plt.annotate('184 mg/dL',
             xy=(arrow_x, arrow_y),
             xytext=(arrow_x + 20, arrow_y + 1),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red')

plt.grid(True)
plt.tight_layout()
plt.show()
