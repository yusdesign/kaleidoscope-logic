#!/usr/bin/env python3
"""
generate_sequences.py
Core script for generating all kaleidoscope sequences
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import json
from datetime import datetime

class SequenceGenerator:
    """
    Generate kaleidoscope sequences with various parameters
    """
    
    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
        self.setup_directories()
        
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            # Sequence parameters
            'sequences': {
                'main': {'frames': 25, 'delta_left': 0.08, 'delta_right': 0.095},
                'rotation': {'frames': 12, 'delta_left': 0.30, 'delta_right': 0.35},
                'quick': {'frames': 10, 'delta_left': 0.15, 'delta_right': 0.18},
            },
            # Visual parameters
            'visual': {
                'resolution': 512,
                'dpi': 100,
                'frame_style': {'bg_color': 'black', 'show_grid': False},
                'exact_style': {'bg_color': 'white', 'show_grid': True},
            },
            # Output directories
            'directories': {
                'sequences': 'outputs/sequences',
                'frames': 'outputs/frames',
                'metadata': 'outputs/metadata',
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Deep merge
                import copy
                config = copy.deepcopy(default_config)
                self._deep_update(config, user_config)
                return config
        return default_config
    
    def _deep_update(self, original, update):
        """Recursively update nested dictionary"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def setup_directories(self):
        """Create all necessary directories"""
        dirs = self.config['directories']
        for dir_path in dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def hsv_to_rgb(self, h, s=0.8, v=0.9):
        """Convert HSV to RGB"""
        h = h % 1.0
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if i == 0: return (v, t, p)
        elif i == 1: return (q, v, p)
        elif i == 2: return (p, v, t)
        elif i == 3: return (p, q, v)
        elif i == 4: return (t, p, v)
        else: return (v, p, q)
    
    def generate_fragments(self, seed=42, num_points=9, pattern='spiral'):
        """Generate fragment positions with different patterns"""
        np.random.seed(seed)
        
        if pattern == 'spiral':
            angles = np.linspace(0, 3*np.pi, num_points)
            radii = np.linspace(0.3, 1.2, num_points)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            
        elif pattern == 'cluster':
            clusters = 3
            points_per_cluster = num_points // clusters
            centers = np.random.uniform(-0.7, 0.7, (clusters, 2))
            points = []
            for i in range(clusters):
                for _ in range(points_per_cluster):
                    point = centers[i] + np.random.normal(0, 0.2, 2)
                    points.append(point)
            points = np.array(points)[:num_points]
            x, y = points[:, 0], points[:, 1]
            
        elif pattern == 'circle':
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            radius = 0.8
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            
        else:  # random
            x = np.random.uniform(-1, 1, num_points)
            y = np.random.uniform(-1, 1, num_points)
        
        return np.column_stack([x, y])
    
    def apply_symmetry(self, fragments, angle, symmetry_type='kaleidoscopic', folds=6):
        """Apply symmetry operations"""
        symmetric_points = []
        
        if symmetry_type == 'kaleidoscopic':
            for point in fragments:
                symmetric_points.append(point)
                for fold in range(1, folds):
                    rot_angle = angle + fold * (2*np.pi / folds)
                    cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
                    x, y = point
                    rot_point = np.array([
                        x * cos_a - y * sin_a,
                        x * sin_a + y * cos_a
                    ])
                    symmetric_points.append(rot_point)
                    symmetric_points.append([-rot_point[0], rot_point[1]])
                    symmetric_points.append([rot_point[0], -rot_point[1]])
                    
        elif symmetry_type == 'rotational':
            for point in fragments:
                symmetric_points.append(point)
                for fold in range(1, folds):
                    rot_angle = angle + fold * (2*np.pi / folds)
                    cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
                    x, y = point
                    symmetric_points.append([
                        x * cos_a - y * sin_a,
                        x * sin_a + y * cos_a
                    ])
        
        return np.array(symmetric_points)
    
    def create_voronoi_background(self, points, bounds=(-2, 4, -2, 4), resolution=150):
        """Create Voronoi-like background pattern"""
        x_min, x_max, y_min, y_max = bounds
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        distances = np.sum((grid_points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2)
        nearest = np.argmin(distances, axis=1)
        
        return xx, yy, nearest.reshape(xx.shape)
    
    def generate_frame(self, frame_num, total_frames, left_angle, right_angle,
                      left_fragments, right_fragments, left_colors, right_colors,
                      style='frame', symmetry_folds=6):
        """Generate a single frame"""
        
        # Apply symmetry
        left_pattern = self.apply_symmetry(left_fragments, left_angle, 
                                         symmetry_type='kaleidoscopic', 
                                         folds=symmetry_folds)
        right_pattern = self.apply_symmetry(right_fragments, right_angle,
                                          symmetry_type='kaleidoscopic',
                                          folds=symmetry_folds)
        
        # Combine patterns
        combined = []
        colors = []
        
        # Left points
        for i, point in enumerate(left_pattern[:len(left_pattern)//2]):
            combined.append(point * 1.2)
            colors.append(left_colors[i % len(left_colors)])
        
        # Right points
        for i, point in enumerate(right_pattern[:len(right_pattern)//2]):
            combined.append(point * 1.2)
            colors.append(right_colors[i % len(right_colors)])
        
        # Fused points
        min_len = min(len(left_pattern)//2, len(right_pattern)//2)
        for i in range(min_len):
            phase = np.sin(right_angle - left_angle + i*0.2)
            fused = (left_pattern[i] * (0.5 + 0.3*phase) + 
                    right_pattern[i] * (0.5 - 0.3*phase))
            combined.append(fused)
            
            # Color fusion
            lc = left_colors[i % len(left_colors)]
            rc = right_colors[i % len(right_colors)]
            mix = (np.sin(phase) + 1) / 2
            fused_color = (
                lc[0] * mix + rc[0] * (1-mix),
                lc[1] * mix + rc[1] * (1-mix),
                lc[2] * mix + rc[2] * (1-mix)
            )
            colors.append(fused_color)
        
        combined = np.array(combined)
        
        # Create figure
        if style == 'exact':
            fig, ax = plt.subplots(figsize=(6.9, 7.17), facecolor='white')
            ax.set_facecolor('white')
            
            # Add grid and labels
            ax.grid(True, color='lightgray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Reality Phase Space', fontsize=10)
            ax.set_ylabel('Perceptual Dimension', fontsize=10)
            ax.set_xlim(-2, 4)
            ax.set_ylim(-2, 4)
            ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
            ax.set_yticks([-2, -1, 0, 1, 2, 3, 4])
            
            title = f'Binocular Kaleidoscope Logic\nLeft θ={left_angle:.2f}, Right θ={right_angle:.2f}'
            ax.set_title(title, fontsize=11)
            
        else:  # frame style
            fig, ax = plt.subplots(figsize=(6.9, 7.17), facecolor='black')
            ax.set_facecolor('black')
            ax.axis('off')
            
            # Add frame info
            ax.text(0.02, 0.98, f'Frame {frame_num+1:02d}/{total_frames}', 
                   transform=ax.transAxes, color='gold', fontsize=12, va='top')
            ax.text(0.98, 0.02, f'θL={left_angle:.2f}  θR={right_angle:.2f}', 
                   transform=ax.transAxes, color='white', fontsize=9, ha='right', va='bottom')
        
        # Create Voronoi background
        xx, yy, voronoi_img = self.create_voronoi_background(combined)
        ax.imshow(voronoi_img, extent=(-2, 4, -2, 4), origin='lower',
                 cmap='hsv', alpha=0.6, interpolation='bilinear')
        
        # Plot points
        ax.scatter(combined[:, 0], combined[:, 1], c=colors[:len(combined)],
                  s=25, alpha=0.9, edgecolors='white', linewidths=0.3)
        
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return fig
    
    def generate_sequence(self, sequence_name='main', style='both', 
                         symmetry_folds=6, fragments_per_eye=9):
        """Generate complete sequence"""
        
        seq_config = self.config['sequences'][sequence_name]
        frames = seq_config['frames']
        delta_left = seq_config['delta_left']
        delta_right = seq_config['delta_right']
        
        # Generate fragments
        left_fragments = self.generate_fragments(42, fragments_per_eye, 'spiral')
        right_fragments = self.generate_fragments(43, fragments_per_eye, 'spiral')
        
        # Generate colors
        left_colors = [self.hsv_to_rgb(i/fragments_per_eye) for i in range(fragments_per_eye)]
        right_colors = [self.hsv_to_rgb(i/fragments_per_eye + 0.15) for i in range(fragments_per_eye)]
        
        # Initial angles
        left_angle = 0.0
        right_angle = 0.39 if sequence_name == 'main' else 0.5
        
        # Determine output styles
        if style == 'both':
            styles = ['frame', 'exact']
        else:
            styles = [style]
        
        metadata = {
            'sequence_name': sequence_name,
            'frames': frames,
            'delta_left': delta_left,
            'delta_right': delta_right,
            'symmetry_folds': symmetry_folds,
            'fragments_per_eye': fragments_per_eye,
            'generated_at': datetime.now().isoformat(),
            'frames_data': []
        }
        
        for style_name in styles:
            output_dir = Path(self.config['directories']['sequences']) / f"{sequence_name}_{style_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Generating {sequence_name} sequence ({style_name} style): {frames} frames...")
            
            for frame in range(frames):
                # Generate frame
                fig = self.generate_frame(
                    frame, frames, left_angle, right_angle,
                    left_fragments, right_fragments,
                    left_colors, right_colors,
                    style=style_name,
                    symmetry_folds=symmetry_folds
                )
                
                # Save frame
                filename = output_dir / f"kaleidoscope_{style_name}_{frame:02d}.png"
                fig.savefig(filename, dpi=self.config['visual']['dpi'], 
                           bbox_inches='tight')
                plt.close(fig)
                
                # Record metadata
                if style_name == 'frame':  # Only record once
                    metadata['frames_data'].append({
                        'frame': frame,
                        'left_angle': float(left_angle),
                        'right_angle': float(right_angle),
                        'phase_diff': float(right_angle - left_angle),
                        'filename': str(filename)
                    })
                
                # Update angles for next frame
                left_angle = (left_angle + delta_left) % (2*np.pi)
                right_angle = (right_angle + delta_right) % (2*np.pi)
                
                # Progress
                if (frame + 1) % 5 == 0 or frame == frames - 1:
                    print(f"  Frame {frame+1}/{frames} complete")
        
        # Save metadata
        metadata_dir = Path(self.config['directories']['metadata'])
        metadata_file = metadata_dir / f"{sequence_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Sequence saved to {output_dir.parent}/")
        print(f"📊 Metadata saved to {metadata_file}")
        
        return metadata

def main():
    parser = argparse.ArgumentParser(description='Generate kaleidoscope sequences')
    parser.add_argument('--sequence', choices=['main', 'rotation', 'quick', 'all'],
                       default='main', help='Sequence to generate')
    parser.add_argument('--style', choices=['frame', 'exact', 'both'],
                       default='both', help='Visual style')
    parser.add_argument('--folds', type=int, default=6,
                       help='Symmetry folds (default: 6)')
    parser.add_argument('--fragments', type=int, default=9,
                       help='Fragments per eye (default: 9)')
    parser.add_argument('--config', type=str,
                       help='Custom config file')
    
    args = parser.parse_args()
    
    generator = SequenceGenerator(args.config)
    
    if args.sequence == 'all':
        sequences = ['main', 'rotation', 'quick']
    else:
        sequences = [args.sequence]
    
    all_metadata = {}
    for seq in sequences:
        metadata = generator.generate_sequence(
            sequence_name=seq,
            style=args.style,
            symmetry_folds=args.folds,
            fragments_per_eye=args.fragments
        )
        all_metadata[seq] = metadata
    
    # Generate summary
    print("\n" + "="*50)
    print("🎉 GENERATION COMPLETE!")
    print("="*50)
    
    for seq_name, meta in all_metadata.items():
        print(f"\n{seq_name.upper()} SEQUENCE:")
        print(f"  Frames: {meta['frames']}")
        print(f"  Δθ per frame: {meta['delta_left']:.3f} (L), {meta['delta_right']:.3f} (R)")
        print(f"  Total Δθ: {meta['frames_data'][-1]['phase_diff']:.3f} rad")
        print(f"  Output: {meta['frames_data'][0]['filename'].rsplit('/', 1)[0]}/")

if __name__ == "__main__":
    main()
