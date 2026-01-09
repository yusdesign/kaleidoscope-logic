# notebooks/analysis.ipynb
"""
# Statistical Analysis Notebook

Analyze patterns and phase relationships.
"""

# Cell 1: Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Cell 2: Load metadata
import json

with open('outputs/metadata/main_metadata.json', 'r') as f:
    metadata = json.load(f)

df = pd.DataFrame(metadata['frames_data'])

# Cell 3: Statistical analysis
print("Phase Statistics:")
print(f"Mean Δθ: {df['phase_diff'].mean():.3f} rad")
print(f"Std Δθ: {df['phase_diff'].std():.3f} rad")
print(f"Min Δθ: {df['phase_diff'].min():.3f} rad")
print(f"Max Δθ: {df['phase_diff'].max():.3f} rad")

# Cell 4: Correlation analysis
correlation = df[['left_angle', 'right_angle', 'phase_diff']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Phase Correlation Matrix')
plt.show()

# Cell 5: Fourier analysis
from scipy.fft import fft

# Analyze phase differences as signal
signal = df['phase_diff'].values
N = len(signal)
T = 1.0  # Sampling interval

yf = fft(signal)
xf = np.fft.fftfreq(N, T)[:N//2]

plt.figure(figsize=(10, 4))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlabel('Frequency (frames⁻¹)')
plt.ylabel('Amplitude')
plt.title('Fourier Analysis of Phase Drift')
plt.grid(True)
plt.show()

# Cell 6: Pattern complexity analysis
def calculate_complexity(phases):
    """Calculate pattern complexity from phases"""
    # Simple complexity metric based on phase relationships
    complexity = np.sin(phases['left_angle']) * np.cos(phases['right_angle'])
    return np.abs(complexity)

df['complexity'] = df.apply(calculate_complexity, axis=1)

plt.figure(figsize=(10, 4))
plt.plot(df['frame'], df['complexity'], 'o-', linewidth=2)
plt.xlabel('Frame')
plt.ylabel('Pattern Complexity')
plt.title('Pattern Complexity Evolution')
plt.grid(True, alpha=0.3)
plt.fill_between(df['frame'], df['complexity'], alpha=0.2)
plt.show()
