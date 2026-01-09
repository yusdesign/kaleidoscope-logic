# notebooks/exploration.ipynb
"""
# Kaleidoscope Logic Exploration Notebook

Explore parameters and generate visualizations interactively.
"""

# Cell 1: Setup
import sys
sys.path.append('..')

from src.kaleidoscope import BinocularKaleidoscope
from src.visualization import KaleidoscopeVisualizer
import matplotlib.pyplot as plt
import numpy as np

# Cell 2: Initialize
config = {
    'left_seed': 42,
    'right_seed': 43,
    'symmetry_folds': 6,
    'fragments_per_eye': 9,
    'delta_left': 0.08,
    'delta_right': 0.095
}

kaleido = BinocularKaleidoscope(config)

# Cell 3: Generate single frame
fig = kaleido.generate_frame(style='exact')
plt.show()

# Cell 4: Interactive parameter exploration
from ipywidgets import interact, FloatSlider, IntSlider

@interact(
    delta_left=FloatSlider(min=0.01, max=0.5, step=0.01, value=0.08),
    delta_right=FloatSlider(min=0.01, max=0.5, step=0.01, value=0.095),
    symmetry_folds=IntSlider(min=3, max=12, step=1, value=6),
    frame=IntSlider(min=0, max=24, step=1, value=0)
)
def explore_parameters(delta_left, delta_right, symmetry_folds, frame):
    kaleido.config.delta_left = delta_left
    kaleido.config.delta_right = delta_right
    kaleido.config.symmetry_folds = symmetry_folds
    
    kaleido.reset_phases()
    for _ in range(frame):
        kaleido.rotate()
    
    fig = kaleido.generate_frame(style='exact')
    plt.show()
    
# Cell 5: Phase analysis
import pandas as pd

# Generate sequence data
kaleido.reset_phases()
data = []
for i in range(25):
    data.append({
        'frame': i,
        'left_phase': kaleido.left_phase,
        'right_phase': kaleido.right_phase,
        'phase_diff': kaleido.right_phase - kaleido.left_phase
    })
    kaleido.rotate()

df = pd.DataFrame(data)
df.plot(x='frame', y=['left_phase', 'right_phase', 'phase_diff'])
plt.title('Phase Evolution')
plt.show()
