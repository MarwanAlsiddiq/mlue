import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate NHANES data for males 40-60 (2021-2022)
np.random.seed(42)
mu, sigma, n = 196, 34, 1500  # CDC reported mean = 196 mg/dL
samples = np.random.normal(mu, sigma, n)
samples = np.clip(samples, 80, 350)  # Apply realistic bounds

# Create bins and calculate distribution
bins = np.arange(50, 405, 5)
hist, bin_edges = np.histogram(samples, bins=bins, density=True)
population_perc = hist * 100 * np.diff(bins)

# Create CSV with distribution data
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
dist_df = pd.DataFrame({
    'cholesterol_level': bin_centers,
    'population_perc': population_perc
})
dist_df.to_csv('cholesterol_distribution.csv', index=False)

# Create line chart with arrow highlighting
plt.figure(figsize=(12, 6))
plt.plot(dist_df['cholesterol_level'], dist_df['population_perc'], 
         color='steelblue', linewidth=2.5, marker='o', markersize=3)

# Highlight the 184 mg/dL level with arrow
highlight_level = 184
highlight_idx = np.abs(dist_df['cholesterol_level'] - highlight_level).idxmin()
arrow_height = dist_df.loc[highlight_idx, 'population_perc'] + 0.15

plt.annotate('Ideal Target (184 mg/dL)', 
             xy=(highlight_level, dist_df.loc[highlight_idx, 'population_perc']),
             xytext=(highlight_level+40, arrow_height+0.5),
             arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=1.5),
             ha='center', color='darkred', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1))

# Add vertical line at 184 mg/dL
plt.axvline(x=highlight_level, color='red', linestyle='--', alpha=0.7)

# Chart formatting
plt.title('Total Cholesterol Distribution: US Males 40-60 (NHANES 2021-2022)', fontsize=14)
plt.xlabel('Total Cholesterol (mg/dL)', fontsize=12)
plt.ylabel('Percentage of Individuals (%)', fontsize=12)
plt.xticks(np.arange(50, 401, 25))
plt.xlim(50, 400)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.fill_between(dist_df['cholesterol_level'], dist_df['population_perc'], 
                 color='steelblue', alpha=0.2)  # Add area under curve

plt.tight_layout()
plt.savefig('cholesterol_line_distribution.png', dpi=300)
plt.show()