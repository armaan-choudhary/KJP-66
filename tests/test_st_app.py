import unittest
import os
from unittest.mock import patch, MagicMock

# Import the function to test
from utils.hardware import get_model_mb

class TestDashboardRenderingLogic(unittest.TestCase):
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_get_model_mb_baseline(self, mock_getsize, mock_exists):
        """Test that get_model_mb returns accurate physical size for baseline."""
        # Setup mock file size to be exactly 64 MB (64 * 1024 * 1024 bytes)
        mock_exists.return_value = True
        mock_getsize.return_value = 64 * (1024 ** 2)
        
        # Test baseline logic
        size = get_model_mb('rtdetr-x.pt')
        
        # Verify it returns exactly 64.0
        self.assertEqual(size, 64.0)
        mock_exists.assert_called_with('rtdetr-x.pt')
        mock_getsize.assert_called_with('rtdetr-x.pt')

    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_get_model_mb_optimised(self, mock_getsize, mock_exists):
        """Test that get_model_mb returns accurate physical size for optimised model."""
        # Setup mock file size to be exactly 34.5 MB
        mock_exists.return_value = True
        mock_getsize.return_value = 34.5 * (1024 ** 2)
        
        # Test optimized logic
        size = get_model_mb('prismnet_optimised.pt')
        
        # Verify it returns exactly 34.5
        self.assertEqual(size, 34.5)
        mock_exists.assert_called_with('prismnet_optimised.pt')
        mock_getsize.assert_called_with('prismnet_optimised.pt')
        
    @patch('os.path.exists')
    def test_get_model_mb_missing_file(self, mock_exists):
        """Test fallback when model file is missing."""
        mock_exists.return_value = False
        
        size = get_model_mb('missing_file.pt')
        
        self.assertEqual(size, 0.0)

if __name__ == '__main__':
    unittest.main()
