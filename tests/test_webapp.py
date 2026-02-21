"""
Unit tests for web application.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_webapp_import():
    """Test that webapp module can be imported."""
    try:
        from src.webapp import app
        assert app is not None
    except ImportError as e:
        pytest.skip(f"Webapp module not available: {e}")


def test_streamlit_available():
    """Test that Streamlit is available."""
    try:
        import streamlit as st
        assert st is not None
    except ImportError:
        pytest.fail("Streamlit not installed")


def test_predict_module_import():
    """Test that prediction module can be imported."""
    try:
        from src.models import predict
        assert predict is not None
    except ImportError as e:
        pytest.fail(f"Prediction module not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
