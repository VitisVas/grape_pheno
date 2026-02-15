#!/usr/bin/env python3
"""
Example/Test script for vascular trait extraction

This demonstrates how to use the process_vasculars module.
"""

from process_vasculars import VascularAnalyzer

# Create analyzer with custom config
analyzer = VascularAnalyzer(
    do_shrink=True
)

# Process image
results = analyzer.process('image_name.png')
print(results.T)
