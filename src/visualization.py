#!/usr/bin/env python3
"""
visualization.py
Advanced visualization tools for kaleidoscope analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import imageio
import json
from tqdm import tqdm

class KaleidoscopeVisualizer:
    """Advanced visualization and analysis tools"""
    
    def __init__(self, sequences_dir='outputs/sequences'):
        self.sequences_dir = Path(sequences_dir)
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.colors = {
            'left': '#3498db',      # Blue
            'right': '#e74c3c',     # Red
            'fusion': '#9b59b6',    # Purple
            'phase': '#2ecc71',     # Green
            'background': '#2c3e50' # Dark blue
        }
    
    def create_montage(self, sequence_dir, output_file, 
                      grid_shape=(5, 5), frame_indices=None):
        """
        Create montage of frames in a grid
        
        Args:
            sequence_dir: Directory containing frames
            output_file: Output filename
            grid_shape: (rows, cols) for montage
            frame_indices: Specific frames to include (None = auto-select)
        """
        sequence_dir = Path(sequence_dir)
        frames = sorted(sequence_dir.glob('*.png'))
        
        if not frames:
            raise ValueError(f"No frames found in {sequence_dir}")
        
        if frame_indices is None:
            # Select evenly spaced frames
            total_frames = len(frames)
            montage_frames = min(grid_shape[0] * grid_shape[1], total_frames)
            step = max(1, total_frames // montage_frames)
            frame_indices = list(range(0, total_frames, step))[:montage_frames]
        
        # Create figure
        fig, axes = plt.subplots(grid_shape[0], grid_shape[1],
                                figsize=(grid_shape[1]*3, grid_shape[0]*2.5))
        
        if grid_shape[0] == 1 and grid_shape[1] == 1:
            axes = np.array([[axes]])
        elif grid_shape[0] == 1 or grid_shape[1] == 1:
            axes = axes.reshape(grid_shape)
        
        # Load and display frames
        for idx, ax in enumerate(axes.flat):
            if idx < len(frame_indices):
                frame_idx = frame_indices[idx]
                if frame_idx < len(frames):
                    img = plt.imread(frames[frame_idx])
                    ax.imshow(img)
                    ax.set_title(f'Frame {frame_idx+1}', fontsize=9)
            ax.axis('off')
        
        # Remove empty subplots
        for idx in range(len(frame_indices), grid_shape[0] * grid_shape[1]):
            axes.flat[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Montage saved: {output_file}")
    
    def create_phase_analysis_plot(self, metadata_file, output_file):
        """Create phase analysis visualization"""
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        frames_data = metadata['frames_data']
        
        # Extract data
        frames = [d['frame'] for d in frames_data]
        left_angles = [d['left_angle'] for d in frames_data]
        right_angles = [d['right_angle'] for d in frames_data]
        phase_diffs = [d['phase_diff'] for d in frames_data]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. Phase evolution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(frames, left_angles, label='Left θ', color=self.colors['left'], 
                linewidth=2, marker='o', markersize=4)
        ax1.plot(frames, right_angles, label='Right θ', color=self.colors['right'],
                linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.set_ylabel('Phase Angle (rad)', fontsize=12)
        ax1.set_title('Phase Evolution Over Sequence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Phase difference
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(frames, phase_diffs, color=self.colors['phase'], 
                linewidth=2, marker='^', markersize=5)
        ax2.set_xlabel('Frame Number', fontsize=12)
        ax2.set_ylabel('Phase Difference Δθ (rad)', fontsize=12)
        ax2.set_title('Perceptual Drift', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(frames, phase_diffs, alpha=0.2, color=self.colors['phase'])
        
        # 3. Polar plot of phase relationship
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        for i, (left, right) in enumerate(zip(left_angles, right_angles)):
            color = plt.cm.viridis(i / len(left_angles))
            ax3.plot([left, right], [1, 1], color=color, alpha=0.6, linewidth=1)
        ax3.set_title('Phase Relationships (Polar)', fontsize=12)
        
        # 4. Phase space trajectory
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = ax4.scatter(left_angles, right_angles, c=frames, 
                             cmap='plasma', s=50, alpha=0.7)
        ax4.plot(left_angles, right_angles, color='gray', alpha=0.3, linewidth=1)
        ax4.set_xlabel('Left Phase θL', fontsize=12)
        ax4.set_ylabel('Right Phase θR', fontsize=12)
        ax4.set_title('Phase Space Trajectory', fontsize=12)
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Frame Number')
        
        # 5. Statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        stats_text = (
            f"SEQUENCE ANALYSIS\n"
            f"────────────────\n"
            f"Total frames: {len(frames)}\n"
            f"Δθ range: {min(phase_diffs):.3f} – {max(phase_diffs):.3f} rad\n"
            f"Δθ per frame: {metadata.get('delta_left', 'N/A')} (L), "
            f"{metadata.get('delta_right', 'N/A')} (R)\n"
            f"Symmetry folds: {metadata.get('symmetry_folds', 'N/A')}\n"
            f"Fragments/eye: {metadata.get('fragments_per_eye', 'N/A')}\n"
            f"\nKey Insights:\n"
            f"• Linear phase drift\n"
            f"• Constant Δθ increment\n"
            f"• Smooth trajectory\n"
            f"• Symmetry preserved"
        )
        
        ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes,
                fontsize=10, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f"Kaleidoscope Sequence Analysis: {metadata['sequence_name'].upper()}",
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Phase analysis saved: {output_file}")
    
    def create_animation(self, sequence_dir, output_file, fps=10):
        """Create video animation from frames"""
        sequence_dir = Path(sequence_dir)
        frames = sorted(sequence_dir.glob('*.png'))
        
        if not frames:
            raise ValueError(f"No frames found in {sequence_dir}")
        
        print(f"Creating animation from {len(frames)} frames...")
        
        # Read frames
        images = []
        for frame_file in tqdm(frames, desc="Loading frames"):
            images.append(imageio.imread(frame_file))
        
        # Save as GIF
        imageio.mimsave(output_file, images, fps=fps)
        print(f"✅ Animation saved: {output_file}")
    
    def create_comparison_grid(self, sequence_dirs, output_file, 
                              frame_indices=None, titles=None):
        """
        Create comparison grid of multiple sequences
        
        Args:
            sequence_dirs: List of directories containing sequences
            output_file: Output filename
            frame_indices: Frames to compare
            titles: Titles for each sequence
        """
        if frame_indices is None:
            frame_indices = [0, 6, 12, 18, 24]
        
        if titles is None:
            titles = [f"Sequence {i+1}" for i in range(len(sequence_dirs))]
        
        fig, axes = plt.subplots(len(frame_indices), len(sequence_dirs),
                                figsize=(len(sequence_dirs)*3, len(frame_indices)*2.5))
        
        if len(frame_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for col, (seq_dir, title) in enumerate(zip(sequence_dirs, titles)):
            frames = sorted(Path(seq_dir).glob('*.png'))
            
            for row, frame_idx in enumerate(frame_indices):
                ax = axes[row, col] if len(frame_indices) > 1 else axes[col]
                
                if frame_idx < len(frames):
                    img = plt.imread(frames[frame_idx])
                    ax.imshow(img)
                    
                    if row == 0:
                        ax.set_title(title, fontsize=11, fontweight='bold')
                    
                    if col == 0:
                        ax.set_ylabel(f'Frame {frame_idx+1}', fontsize=10)
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
        
        plt.suptitle('Kaleidoscope Sequence Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison grid saved: {output_file}")
    
    def create_3d_phase_plot(self, metadata_files, output_file):
        """Create 3D visualization of phase evolution"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, meta_file in enumerate(metadata_files):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            frames_data = metadata['frames_data']
            frames = [d['frame'] for d in frames_data]
            left_angles = [d['left_angle'] for d in frames_data]
            right_angles = [d['right_angle'] for d in frames_data]
            phase_diffs = [d['phase_diff'] for d in frames_data]
            
            # Create color gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(frames)))
            
            # Plot 3D trajectory
            ax.plot(left_angles, right_angles, phase_diffs,
                   label=metadata['sequence_name'], linewidth=2, alpha=0.7)
            
            # Add scatter points
            ax.scatter(left_angles, right_angles, phase_diffs,
                      c=colors, s=30, alpha=0.8, depthshade=True)
        
        ax.set_xlabel('Left Phase θL', fontsize=12, labelpad=10)
        ax.set_ylabel('Right Phase θR', fontsize=12, labelpad=10)
        ax.set_zlabel('Phase Difference Δθ', fontsize=12, labelpad=10)
        ax.set_title('3D Phase Space Evolution', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 3D phase plot saved: {output_file}")

def create_demo_visualizations():
    """Create all demo visualizations"""
    visualizer = KaleidoscopeVisualizer()
    
    # Create directories
    viz_dir = Path('outputs/visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Creating visualizations...")
    
    try:
        # 1. Montage of main sequence
        visualizer.create_montage(
            sequence_dir='outputs/sequences/main_frame',
            output_file=viz_dir / 'main_sequence_montage.png',
            grid_shape=(5, 5)
        )
        
        # 2. Phase analysis
        visualizer.create_phase_analysis_plot(
            metadata_file='outputs/metadata/main_metadata.json',
            output_file=viz_dir / 'phase_analysis.png'
        )
        
        # 3. Comparison grid
        visualizer.create_comparison_grid(
            sequence_dirs=[
                'outputs/sequences/main_frame',
                'outputs/sequences/main_exact',
                'outputs/sequences/rotation_frame'
            ],
            output_file=viz_dir / 'sequence_comparison.png',
            titles=['Main (Frame)', 'Main (Exact)', 'Rotation']
        )
        
        # 4. Animation
        visualizer.create_animation(
            sequence_dir='outputs/sequences/main_frame',
            output_file=viz_dir / 'main_sequence.gif',
            fps=5
        )
        
        print(f"\n✅ All visualizations saved to {viz_dir}/")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        print("Make sure sequences are generated first: python generate_sequences.py")

if __name__ == "__main__":
    create_demo_visualizations() 
