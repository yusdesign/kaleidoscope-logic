import unittest
import numpy as np
from src.kaleidoscope import BinocularKaleidoscope

class TestKaleidoscope(unittest.TestCase):
    
    def setUp(self):
        self.kaleido = BinocularKaleidoscope()
    
    def test_initialization(self):
        self.assertEqual(self.kaleido.config.fragments_per_eye, 9)
        self.assertEqual(self.kaleido.config.symmetry_folds, 6)
    
    def test_phase_rotation(self):
        initial_left = self.kaleido.left_phase
        initial_right = self.kaleido.right_phase
        
        self.kaleido.rotate()
        
        self.assertNotEqual(self.kaleido.left_phase, initial_left)
        self.assertNotEqual(self.kaleido.right_phase, initial_right)
        
        # Check modulo 2π
        self.assertLess(self.kaleido.left_phase, 2*np.pi)
        self.assertLess(self.kaleido.right_phase, 2*np.pi)
    
    def test_symmetry_application(self):
        fragments = np.array([[1, 0], [0, 1]])
        symmetric = self.kaleido._apply_symmetry(fragments, 0)
        
        # Should have more points than input
        self.assertGreater(len(symmetric), len(fragments))
        
        # All points should be 2D
        self.assertEqual(symmetric.shape[1], 2)

if __name__ == '__main__':
    unittest.main() 
