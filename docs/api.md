# Code documentation

📚 Complete API Documentation

API Documentation: Kaleidoscope Logic Framework

Table of Contents

1. Overview
2. Installation
3. Core Classes
4. Sequence Generation
5. Visualization
6. Analysis
7. Configuration
8. Command Line Interface
9. Examples
10. Advanced Usage

Overview

The Kaleidoscope Logic Framework provides tools for generating, visualizing, and analyzing binocular kaleidoscope patterns that model constructed perception through symmetry operations.

Installation

From Source

```bash
git clone https://github.com/yusdesign/kaleidoscope-logic.git
cd kaleidoscope-logic
pip install -r requirements.txt
```

As Package (Future)

```bash
pip install kaleidoscope-logic
```

Core Classes

BinocularKaleidoscope

Main class for generating kaleidoscope patterns.

```python
from src.kaleidoscope import BinocularKaleidoscope, KaleidoscopeConfig

# With default configuration
kaleido = BinocularKaleidoscope()

# With custom configuration
config = KaleidoscopeConfig(
    left_seed=42,
    right_seed=43,
    symmetry_folds=6,
    fragments_per_eye=9,
    delta_left=0.08,
    delta_right=0.095,
    initial_left=0.0,
    initial_right=0.39,
    resolution=512
)
kaleido = BinocularKaleidoscope(config)
```

Configuration Parameters

| Parameter | Type | Default | Description |  
| --- | --- | --- | --- |  
| left_seed | int | 42 | Random seed for left fragments |  
| right_seed | int | 43 | Random seed for right fragments |
| symmetry_folds | int | 6 | Number of symmetry folds (mirrors × 2) |  
| fragments_per_eye | int | 9 | Number of initial fragments per system |  
| delta_left | float | 0.08 | Left phase increment per frame (rad) |  
| delta_right | float | 0.095 | Right phase increment per frame (rad) |  
| initial_left | float | 0.0 | Initial left phase (rad) |  
| initial_right | float | 0.39 | Initial right phase (rad) |
| color_palette | str | 'philosophical' | Color scheme ('philosophical', 'complementary', 'rainbow') |  
| resolution | int | 512 | Image resolution |  
| dpi | int | 100 | Image DPI for saving |  

Methods

reset_phases()

```python
kaleido.reset_phases()
# Resets phase angles to initial values
```

rotate()

```python
kaleido.rotate()
# Advances phases by one step (delta_left, delta_right)
```

generate_frame(style='frame')

```python
# Generate artistic frame (black background)
fig = kaleido.generate_frame(style='frame')

# Generate scientific frame (white background with grid)
fig = kaleido.generate_frame(style='exact')
```

generate_sequence(num_frames=25, output_dir=None, style='frame', prefix='frame')

```python
# Generate 25-frame artistic sequence
files = kaleido.generate_sequence(
    num_frames=25,
    output_dir="outputs/sequences",
    style='frame',
    prefix='kaleidoscope_frame'
)

# Generate scientific sequence
files = kaleido.generate_sequence(
    num_frames=25,
    output_dir="outputs/sequences",
    style='exact', 
    prefix='kaleidoscope_exact'
)
```

SequenceGenerator

Advanced sequence generation with multiple patterns.

```python
from src.generate_sequences import SequenceGenerator

# Initialize with default config
generator = SequenceGenerator()

# Or with custom config file
generator = SequenceGenerator('configs/custom.json')
```

Methods

generate_sequence(sequence_name='main', style='both', symmetry_folds=6, fragments_per_eye=9)

```python
# Generate main sequence with both styles
metadata = generator.generate_sequence(
    sequence_name='main',
    style='both',  # 'frame', 'exact', or 'both'
    symmetry_folds=6,
    fragments_per_eye=9
)

# Generate rotation sequence
metadata = generator.generate_sequence(
    sequence_name='rotation',
    style='frame',
    symmetry_folds=6
)

# Generate quick demo
metadata = generator.generate_sequence(
    sequence_name='quick',
    style='exact',
    symmetry_folds=8
)
```

Available Sequences:

Sequence Frames ΔL ΔR Description
main 25 0.08 0.095 Full 25-frame evolution
rotation 12 0.30 0.35 12-step rotation analysis
quick 10 0.15 0.18 Quick demonstration

KaleidoscopeVisualizer

Advanced visualization and analysis tools.

```python
from src.visualization import KaleidoscopeVisualizer

visualizer = KaleidoscopeVisualizer(sequences_dir='outputs/sequences')
```

Methods

create_montage(sequence_dir, output_file, grid_shape=(5, 5), frame_indices=None)

```python
# Create montage of frames
visualizer.create_montage(
    sequence_dir='outputs/sequences/main_frame',
    output_file='outputs/visualizations/main_montage.png',
    grid_shape=(5, 5),  # 5x5 grid
    frame_indices=None  # Auto-select evenly spaced
)

# Specific frames
visualizer.create_montage(
    sequence_dir='outputs/sequences/main_frame',
    output_file='outputs/visualizations/key_frames.png',
    grid_shape=(2, 3),
    frame_indices=[0, 6, 12, 18, 24]
)
```

create_phase_analysis_plot(metadata_file, output_file)

```python
# Create phase analysis visualization
visualizer.create_phase_analysis_plot(
    metadata_file='outputs/metadata/main_metadata.json',
    output_file='outputs/visualizations/phase_analysis.png'
)
```

create_animation(sequence_dir, output_file, fps=10)

```python
# Create GIF animation
visualizer.create_animation(
    sequence_dir='outputs/sequences/main_frame',
    output_file='outputs/visualizations/main_sequence.gif',
    fps=5  # Frames per second
)

# Create MP4 video (requires ffmpeg)
visualizer.create_animation(
    sequence_dir='outputs/sequences/main_frame',
    output_file='outputs/visualizations/main_sequence.mp4',
    fps=24
)
```

create_comparison_grid(sequence_dirs, output_file, frame_indices=None, titles=None)

```python
# Compare multiple sequences
visualizer.create_comparison_grid(
    sequence_dirs=[
        'outputs/sequences/main_frame',
        'outputs/sequences/main_exact',
        'outputs/sequences/rotation_frame'
    ],
    output_file='outputs/visualizations/comparison.png',
    frame_indices=[0, 6, 12, 18, 24],
    titles=['Main (Frame)', 'Main (Exact)', 'Rotation']
)
```

create_3d_phase_plot(metadata_files, output_file)

```python
# 3D visualization of phase evolution
visualizer.create_3d_phase_plot(
    metadata_files=[
        'outputs/metadata/main_metadata.json',
        'outputs/metadata/rotation_metadata.json'
    ],
    output_file='outputs/visualizations/3d_phase_plot.png'
)
```

Sequence Generation

Using Command Line

```bash
# Generate main sequence with both styles
python generate_sequences.py --sequence main --style both

# Generate rotation sequence with frame style only
python generate_sequences.py --sequence rotation --style frame

# Generate all sequences
python generate_sequences.py --sequence all

# Custom parameters
python generate_sequences.py --sequence main --style both --folds 8 --fragments 12

# With custom config
python generate_sequences.py --sequence main --config configs/custom.json
```

Programmatic Generation

```python
from src.generate_sequences import SequenceGenerator

# Generate complete set for paper
generator = SequenceGenerator()

# Generate all sequences
sequences = ['main', 'rotation', 'quick']
all_metadata = {}

for seq in sequences:
    metadata = generator.generate_sequence(
        sequence_name=seq,
        style='both',
        symmetry_folds=6,
        fragments_per_eye=9
    )
    all_metadata[seq] = metadata

# Access metadata
print(f"Main sequence: {all_metadata['main']['frames']} frames")
print(f"Phase range: {all_metadata['main']['frames_data'][-1]['phase_diff']:.3f} rad")
```

Visualization

Quick Visualization

```python
from src.visualization import create_demo_visualizations

# Create all standard visualizations
create_demo_visualizations()
# Creates:
# - outputs/visualizations/main_sequence_montage.png
# - outputs/visualizations/phase_analysis.png  
# - outputs/visualizations/sequence_comparison.png
# - outputs/visualizations/main_sequence.gif
```

Custom Visualization Pipeline

```python
from src.visualization import KaleidoscopeVisualizer

visualizer = KaleidoscopeVisualizer()

# 1. Create detailed phase analysis
visualizer.create_phase_analysis_plot(
    'outputs/metadata/main_metadata.json',
    'outputs/visualizations/detailed_phase_analysis.png'
)

# 2. Create comparison with different parameters
visualizer.create_comparison_grid(
    sequence_dirs=[
        'outputs/sequences/main_frame',
        'outputs/sequences/rotation_frame',
        'outputs/sequences/quick_frame'
    ],
    output_file='outputs/visualizations/parameter_comparison.png',
    titles=['Main (ΔL=0.08)', 'Rotation (ΔL=0.30)', 'Quick (ΔL=0.15)']
)

# 3. Create animation with custom FPS
visualizer.create_animation(
    'outputs/sequences/main_frame',
    'outputs/visualizations/slow_animation.gif',
    fps=2  # Slow animation for contemplation
)
```

Analysis

Statistical Analysis

```python
import pandas as pd
import json

# Load metadata
with open('outputs/metadata/main_metadata.json', 'r') as f:
    metadata = json.load(f)

df = pd.DataFrame(metadata['frames_data'])

# Calculate statistics
print(f"Total frames: {len(df)}")
print(f"Mean phase difference: {df['phase_diff'].mean():.3f} rad")
print(f"Phase range: {df['phase_diff'].min():.3f} - {df['phase_diff'].max():.3f} rad")
print(f"Phase increment per frame: {metadata['delta_right'] - metadata['delta_left']:.3f} rad")

# Calculate complexity (example metric)
df['complexity'] = np.abs(np.sin(df['left_angle']) * np.cos(df['right_angle']))
print(f"Peak complexity: {df['complexity'].max():.3f} at frame {df['complexity'].idxmax()}")
```

Fourier Analysis

```python
from scipy.fft import fft
import numpy as np

# Analyze phase difference as signal
signal = df['phase_diff'].values
N = len(signal)

# Compute FFT
yf = fft(signal)
xf = np.fft.fftfreq(N, 1.0)[:N//2]
power_spectrum = 2.0/N * np.abs(yf[0:N//2])

# Find dominant frequency
dominant_idx = np.argmax(power_spectrum[1:]) + 1
dominant_freq = xf[dominant_idx]

print(f"Dominant frequency: {dominant_freq:.3f} frame⁻¹")
print(f"Dominant period: {1/dominant_freq:.1f} frames")
```

Configuration

Configuration Files

Create JSON configuration files in configs/:

```json
{
  "sequences": {
    "custom": {
      "frames": 36,
      "delta_left": 0.05,
      "delta_right": 0.06,
      "description": "Slow evolution for meditation"
    }
  },
  "visual": {
    "resolution": 1024,
    "dpi": 150,
    "frame_style": {
      "bg_color": "darkblue",
      "show_grid": false,
      "show_labels": false
    },
    "exact_style": {
      "bg_color": "ivory",
      "show_grid": true,
      "show_labels": true,
      "grid_color": "#cccccc",
      "grid_alpha": 0.2
    }
  },
  "directories": {
    "sequences": "my_sequences",
    "frames": "my_frames",
    "metadata": "my_metadata"
  }
}
```

Environment Configuration

```python
import os
from src.kaleidoscope import KaleidoscopeConfig

# Configure via environment variables
os.environ['KALEIDOSCOPE_RESOLUTION'] = '1024'
os.environ['KALEIDOSCOPE_STYLE'] = 'exact'

config = KaleidoscopeConfig.from_env()
# Automatically reads from environment
```

Command Line Interface

Complete Workflow

```bash
# 1. Setup project
./setup_git.sh

# 2. Generate all sequences
python generate_sequences.py --sequence all --style both

# 3. Create visualizations
python visualization.py

# 4. Run analysis notebook
jupyter notebook notebooks/analysis.ipynb

# 5. Update paper with new figures
./update_paper.sh

# 6. Sync with Overleaf
./sync_overleaf.sh
```

Utility Scripts

```bash
# Generate paper figures only
python src/kaleidoscope.py  # Calls generate_paper_figures()

# Clean generated files
./workflow.sh clean

# Run tests
python -m pytest tests/

# Create API documentation
pdoc src --html --output-dir docs/api
```

Examples

Example 1: Complete Research Pipeline

```python
from src.kaleidoscope import generate_paper_figures
from src.visualization import create_demo_visualizations

# Generate all figures for paper
generate_paper_figures()
print("✅ Generated 50 frames for paper")

# Create analysis visualizations
create_demo_visualizations()
print("✅ Created analysis visualizations")

# Load and analyze metadata
import json
with open('outputs/metadata/main_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\nResearch Summary:")
print(f"Sequence: {metadata['sequence_name']}")
print(f"Frames: {metadata['frames']}")
print(f"Δθ range: {metadata['frames_data'][0]['phase_diff']:.3f} - {metadata['frames_data'][-1]['phase_diff']:.3f} rad")
print(f"Symmetry: {metadata['symmetry_folds']}-fold")
```

Example 2: Interactive Exploration

```python
from src.kaleidoscope import BinocularKaleidoscope
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

kaleido = BinocularKaleidoscope()

@interact(
    left_phase=FloatSlider(min=0, max=6.28, step=0.1, value=0.0),
    right_phase=FloatSlider(min=0, max=6.28, step=0.1, value=0.39),
    style=['frame', 'exact']
)
def explore_phases(left_phase, right_phase, style):
    """Interactive phase exploration"""
    kaleido.left_phase = left_phase
    kaleido.right_phase = right_phase
    
    fig = kaleido.generate_frame(style=style)
    plt.show()
    
    print(f"Phase Analysis:")
    print(f"  Left θ: {left_phase:.2f} rad ({left_phase/np.pi:.2f}π)")
    print(f"  Right θ: {right_phase:.2f} rad ({right_phase/np.pi:.2f}π)")
    print(f"  Δθ: {right_phase-left_phase:.2f} rad")
    print(f"  Coherence: {np.cos(right_phase-left_phase):.2f}")
```

Example 3: Batch Processing

```python
from src.generate_sequences import SequenceGenerator
from src.visualization import KaleidoscopeVisualizer
import pandas as pd

# Generate multiple sequences with different parameters
generator = SequenceGenerator()
visualizer = KaleidoscopeVisualizer()

parameter_sets = [
    {'symmetry_folds': 4, 'fragments': 6},
    {'symmetry_folds': 6, 'fragments': 9},
    {'symmetry_folds': 8, 'fragments': 12},
    {'symmetry_folds': 12, 'fragments': 18}
]

results = []

for i, params in enumerate(parameter_sets):
    print(f"Generating sequence {i+1}/{len(parameter_sets)}...")
    
    metadata = generator.generate_sequence(
        sequence_name=f'experiment_{i}',
        style='both',
        symmetry_folds=params['symmetry_folds'],
        fragments_per_eye=params['fragments']
    )
    
    # Create visualizations
    visualizer.create_montage(
        f"outputs/sequences/experiment_{i}_frame",
        f"outputs/visualizations/experiment_{i}_montage.png"
    )
    
    # Record results
    results.append({
        'experiment': i,
        'symmetry_folds': params['symmetry_folds'],
        'fragments': params['fragments'],
        'final_phase_diff': metadata['frames_data'][-1]['phase_diff'],
        'complexity_std': pd.DataFrame(metadata['frames_data'])['phase_diff'].std()
    })

# Analyze results
results_df = pd.DataFrame(results)
print(results_df)
```

Advanced Usage

Custom Symmetry Operations

```python
class CustomKaleidoscope(BinocularKaleidoscope):
    """Custom kaleidoscope with different symmetry"""
    
    def _apply_symmetry(self, fragments, angle):
        """Custom symmetry operation (e.g., fractal symmetry)"""
        symmetric_points = []
        
        # Base symmetry
        symmetric_points.extend(fragments)
        
        # Fractal levels
        for level in range(3):
            scale = 0.7 ** level
            for point in fragments:
                scaled = point * scale
                symmetric_points.append(scaled)
                
                # Apply rotation
                for fold in range(1, self.config.symmetry_folds):
                    rot_angle = angle + fold * (2*np.pi / self.config.symmetry_folds)
                    cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
                    x, y = scaled
                    rot_point = np.array([
                        x * cos_a - y * sin_a,
                        x * sin_a + y * cos_a
                    ])
                    symmetric_points.append(rot_point)
        
        return np.array(symmetric_points)
```

Real-time Animation

```python
import matplotlib.animation as animation
from src.kaleidoscope import BinocularKaleidoscope

kaleido = BinocularKaleidoscope()

fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    ax.clear()
    fig = kaleido.generate_frame(style='frame')
    # Extract and display frame
    kaleido.rotate()
    return []

ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False)
plt.show()

# Save animation
ani.save('real_time_animation.mp4', writer='ffmpeg', fps=10)
```

Integration with Other Libraries

```python
import plotly.graph_objects as go
from src.kaleidoscope import BinocularKaleidoscope

# Create interactive 3D visualization
kaleido = BinocularKaleidoscope()

# Generate phase data
phases = []
for _ in range(25):
    phases.append({
        'left': kaleido.left_phase,
        'right': kaleido.right_phase,
        'diff': kaleido.right_phase - kaleido.left_phase
    })
    kaleido.rotate()

# Create Plotly figure
fig = go.Figure(data=[
    go.Scatter3d(
        x=[p['left'] for p in phases],
        y=[p['right'] for p in phases],
        z=[p['diff'] for p in phases],
        mode='markers+lines',
        marker=dict(size=5, color=list(range(25)), colorscale='Viridis')
    )
])

fig.update_layout(
    title='3D Phase Space Evolution',
    scene=dict(
        xaxis_title='Left Phase θL',
        yaxis_title='Right Phase θR',
        zaxis_title='Phase Difference Δθ'
    )
)

fig.show()
```

Troubleshooting

Common Issues

1. Missing Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install specific missing packages
pip install numpy matplotlib imageio tqdm pillow
```

1. Image Saving Issues

```python
# Increase DPI for higher quality
config = KaleidoscopeConfig(dpi=300)
kaleido = BinocularKaleidoscope(config)

# Or specify when saving
fig.savefig('output.png', dpi=300, bbox_inches='tight')
```

1. Memory Issues with Large Sequences

```python
# Generate frames in batches
batch_size = 10
total_frames = 100

for batch_start in range(0, total_frames, batch_size):
    batch_end = min(batch_start + batch_size, total_frames)
    frames = kaleido.generate_sequence(
        num_frames=batch_end - batch_start,
        output_dir=f'outputs/batch_{batch_start//batch_size}'
    )
```

1. Color Issues

```python
# Use different color palettes
config = KaleidoscopeConfig(color_palette='complementary')
kaleido = BinocularKaleidoscope(config)

# Or customize colors manually
kaleido.left_colors = [(0.2, 0.4, 0.8) for _ in range(9)]  # Blue
kaleido.right_colors = [(0.8, 0.2, 0.2) for _ in range(9)]  # Red
```

Contributing

Development Setup

```bash
# Clone repository
git clone https://github.com/yusdesign/kaleidoscope-logic.git
cd kaleidoscope-logic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Build documentation
pdoc src --html --output-dir docs/api
```

Adding New Features

1. Create feature branch
2. Add tests in tests/
3. Update documentation in docs/
4. Submit pull request

Citation

If using this framework in research:

```bibtex
@software{kaleidoscope_logic_2024,
  title = {Binocular Kaleidoscope Logic Framework},
  author = {Research Team},
  year = {2024},
  url = {https://github.com/yusdesign/kaleidoscope-logic},
  version = {1.0.0}
}
```
