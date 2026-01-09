# src/kaleidoscope.py
"""
Core kaleidoscope logic implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os

@dataclass
class KaleidoscopeConfig:
    """Configuration for kaleidoscope system"""
    left_seed: int = 42
    right_seed: int = 43
    symmetry_folds: int = 6
    fragments_per_eye: int = 9
    color_palette: str = 'philosophical'
    
    # Phase evolution
    delta_left: float = 0.08
    delta_right: float = 0.095
    initial_left: float = 0.0
    initial_right: float = 0.39
    
    # Visualization
    resolution: int = 512
    background_color: str = 'black'
    dpi: int = 100

class BinocularKaleidoscope:
    """
    Main kaleidoscope system implementing binocular logic
    """
    
    def __init__(self, config: Optional[KaleidoscopeConfig] = None):
        self.config = config or KaleidoscopeConfig()
        self._init_fragments()
        self._init_colors()
        self.reset_phases()
        
    def _init_fragments(self):
        """Initialize fragment positions"""
        np.random.seed(self.config.left_seed)
        angles = np.linspace(0, 2*np.pi, self.config.fragments_per_eye, endpoint=False)
        radii = np.linspace(0.3, 1.0, self.config.fragments_per_eye)
        
        self.left_fragments = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        
        np.random.seed(self.config.right_seed)
        angles = np.linspace(np.pi/9, 2*np.pi + np.pi/9, 
                           self.config.fragments_per_eye, endpoint=False)
        radii = np.linspace(0.35, 0.95, self.config.fragments_per_eye)
        
        self.right_fragments = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
    
    def _init_colors(self):
        """Initialize color schemes"""
        if self.config.color_palette == 'philosophical':
            # Left: rational/cool, Right: emotional/warm
            left_hues = np.linspace(0.5, 0.7, self.config.fragments_per_eye)
            right_hues = np.linspace(0.0, 0.2, self.config.fragments_per_eye)
        else:
            left_hues = np.linspace(0, 1, self.config.fragments_per_eye)
            right_hues = np.linspace(0.2, 0.8, self.config.fragments_per_eye)
        
        self.left_colors = self._hsv_to_rgb_array(left_hues)
        self.right_colors = self._hsv_to_rgb_array(right_hues)
    
    def _hsv_to_rgb_array(self, h_array):
        """Convert HSV array to RGB"""
        rgb_array = []
        for h in h_array:
            h = h % 1.0
            i = int(h * 6)
            f = h * 6 - i
            p = 0.9 * (1 - 0.8)
            q = 0.9 * (1 - f * 0.8)
            t = 0.9 * (1 - (1 - f) * 0.8)
            
            if i == 0: rgb = (0.9, t, p)
            elif i == 1: rgb = (q, 0.9, p)
            elif i == 2: rgb = (p, 0.9, t)
            elif i == 3: rgb = (p, q, 0.9)
            elif i == 4: rgb = (t, p, 0.9)
            else: rgb = (0.9, p, q)
            
            rgb_array.append(rgb)
        return rgb_array
    
    def reset_phases(self):
        """Reset phase angles to initial values"""
        self.left_phase = self.config.initial_left
        self.right_phase = self.config.initial_right
        self.frame_history = []
    
    def rotate(self):
        """Advance phase angles by one step"""
        self.left_phase = (self.left_phase + self.config.delta_left) % (2 * np.pi)
        self.right_phase = (self.right_phase + self.config.delta_right) % (2 * np.pi)
        
        self.frame_history.append({
            'frame': len(self.frame_history),
            'left_phase': self.left_phase,
            'right_phase': self.right_phase,
            'phase_diff': self.right_phase - self.left_phase
        })
    
    def generate_frame(self, style='frame'):
        """
        Generate a single frame
        
        Args:
            style: 'frame' (artistic) or 'exact' (scientific)
        
        Returns:
            matplotlib figure object
        """
        if style == 'exact':
            return self._generate_exact_frame()
        else:
            return self._generate_frame_style()
    
    def _generate_frame_style(self):
        """Generate artistic frame (black background)"""
        fig, ax = plt.subplots(figsize=(6.9, 7.17), facecolor='black')
        ax.set_facecolor('black')
        
        # Generate and plot pattern
        pattern = self._generate_pattern()
        self._plot_pattern(ax, pattern)
        
        # Add frame info
        ax.text(0.02, 0.98, f'Frame {len(self.frame_history)}/25', 
                transform=ax.transAxes, color='gold', fontsize=12, va='top')
        ax.text(0.98, 0.02, f'θL={self.left_phase:.2f}  θR={self.right_phase:.2f}', 
                transform=ax.transAxes, color='white', fontsize=9, ha='right', va='bottom')
        
        ax.axis('off')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return fig
    
    def _generate_exact_frame(self):
        """Generate scientific frame (white background with grid)"""
        fig, ax = plt.subplots(figsize=(6.9, 7.17), facecolor='white')
        
        # Generate and plot pattern
        pattern = self._generate_pattern()
        self._plot_pattern(ax, pattern)
        
        # Scientific formatting
        ax.set_title(f'Binocular Kaleidoscope Logic\nLeft θ={self.left_phase:.2f}, Right θ={self.right_phase:.2f}')
        ax.set_xlabel('Reality Phase Space')
        ax.set_ylabel('Perceptual Dimension')
        ax.grid(True, color='lightgray', linestyle='-', alpha=0.3)
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)
        ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
        ax.set_yticks([-2, -1, 0, 1, 2, 3, 4])
        
        plt.tight_layout()
        return fig
    
    def _generate_pattern(self):
        """Generate the visual pattern (common to both styles)"""
        # Apply symmetry operations
        left_pattern = self._apply_symmetry(self.left_fragments, self.left_phase)
        right_pattern = self._apply_symmetry(self.right_fragments, self.right_phase)
        
        # Combine patterns
        combined = []
        colors = []
        
        # Add points with appropriate colors
        for i, point in enumerate(left_pattern):
            combined.append(point)
            colors.append(self.left_colors[i % len(self.left_colors)])
        
        for i, point in enumerate(right_pattern):
            combined.append(point + np.array([1.5, 0]))
            colors.append(self.right_colors[i % len(self.right_colors)])
        
        # Add fused points
        min_len = min(len(left_pattern), len(right_pattern))
        for i in range(min_len):
            fused = (left_pattern[i] + right_pattern[i]) / 2
            combined.append(fused + np.array([0.75, 1.5]))
            
            # Blend colors
            lc = self.left_colors[i % len(self.left_colors)]
            rc = self.right_colors[i % len(self.right_colors)]
            fused_color = tuple((a + b) / 2 for a, b in zip(lc, rc))
            colors.append(fused_color)
        
        return np.array(combined), colors
    
    def _apply_symmetry(self, fragments, angle):
        """Apply kaleidoscope symmetry"""
        symmetric = []
        for point in fragments:
            symmetric.append(point)
            for fold in range(1, self.config.symmetry_folds):
                rot_angle = angle + fold * (2 * np.pi / self.config.symmetry_folds)
                cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
                x, y = point
                rot_point = np.array([
                    x * cos_a - y * sin_a,
                    x * sin_a + y * cos_a
                ])
                symmetric.append(rot_point)
                symmetric.append([-rot_point[0], rot_point[1]])
                symmetric.append([rot_point[0], -rot_point[1]])
        return np.array(symmetric)
    
    def _plot_pattern(self, ax, pattern):
        """Plot pattern on axis"""
        points, colors = pattern
        
        # Simple Voronoi-like effect
        from scipy.spatial import Voronoi
        vor = Voronoi(points)
        
        # Plot Voronoi regions
        for region in vor.regions:
            if not region or -1 in region:
                continue
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color='gray', alpha=0.1)
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], 
                  c=colors[:len(points)], s=30, alpha=0.9,
                  edgecolors='white', linewidth=0.5)
    
    def generate_sequence(self, num_frames=25, output_dir=None, 
                         style='frame', prefix='frame'):
        """
        Generate sequence of frames
        
        Args:
            num_frames: Number of frames to generate
            output_dir: Output directory
            style: 'frame' or 'exact'
            prefix: Filename prefix
        
        Returns:
            List of saved filenames
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.reset_phases()
        saved_files = []
        
        for i in range(num_frames):
            fig = self.generate_frame(style)
            
            if output_dir:
                filename = f"{output_dir}/{prefix}_{i:02d}.png"
                fig.savefig(filename, dpi=self.config.dpi, 
                           bbox_inches='tight')
                saved_files.append(filename)
            
            plt.close(fig)
            self.rotate()
        
        return saved_files

# Convenience function
def generate_paper_figures():
    """Generate all figures needed for the paper"""
    config = KaleidoscopeConfig(
        delta_left=0.08,
        delta_right=0.095,
        initial_right=0.39
    )
    
    kaleido = BinocularKaleidoscope(config)
    
    # Generate both sequences
    print("Generating frame sequence (artistic style)...")
    kaleido.generate_sequence(
        num_frames=25,
        output_dir="paper/figures/exact_sequence",
        style='frame',
        prefix='kaleidoscope_frame'
    )
    
    print("Generating exact sequence (scientific style)...")
    kaleido.generate_sequence(
        num_frames=25,
        output_dir="paper/figures/exact_sequence",
        style='exact',
        prefix='kaleidoscope_exact'
    )
    
    print("✅ Generated 50 frames for paper")
