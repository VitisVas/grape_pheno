# Vascular Trait Extraction Tool

Python tool for extracting phenotypic traits from vascular networks in fruit slice images.

## Overview

This tool automatically analyzes vascular networks in color-coded fruit slice images and extracts 15 quantitative traits including area measurements, vessel density, branching characteristics, and fractal dimensions.

## Features

- **Automated extraction**: Processes images with minimal user input
- **Comprehensive metrics**: 15 different vascular traits
- **Fast processing**: 5-30 seconds per image
- **Flexible output**: CSV format for easy integration
- **Optional preprocessing**: Edge shrinking and skeleton visualization
- **Silent mode**: Runs quietly by default

## Installation

### Requirements

- Python 3.7 or higher
- pip package manager

### Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy opencv-python scipy scikit-image pandas
```

## Usage

### Command Line

```bash
python process_vasculars.py <input_image> [output_csv] [options]
```

**Options:**
- `--shrink`: Enable edge shrinking preprocessing
- `--debug`: Save skeleton visualization image
- `--verbose`: Show processing output

**Examples:**

```bash
# Basic usage (silent mode, CSV saved as input_filename.csv)
python process_vasculars.py sample.png

# Specify output CSV
python process_vasculars.py sample.png results.csv

# With edge shrinking enabled
python process_vasculars.py sample.png --shrink

# With debug visualization
python process_vasculars.py sample.png --debug

# Verbose mode with all options
python process_vasculars.py sample.png results.csv --shrink --debug --verbose
```

**Bash processing (under Linux bash):**
Automatically process all images found within the current folder. Note: the process_vasculars.py needs to be wihtin the same folder.

```bash
#! /bin/bash

for f in *.png; do
	python process_vasculars.py $f
done

```

### Python API

```python
from process_vasculars import VascularAnalyzer, process_vasculars

# Simple processing (silent mode)
df = process_vasculars('sample.png')

# With options
df = process_vasculars('sample.png', 
                       do_shrink=True,
                       save_debug_image=True,
                       quiet=False)

# Using the class directly
analyzer = VascularAnalyzer(
    border_width_mm=0.75,
    shrinking_distance_mm=0.5,
    branch_merge_distance=20,
    do_shrink=False,
    save_debug_image=False,
    quiet=True
)
results = analyzer.process('sample.png', 'output.csv')

# Access individual traits
print(f"Slice area: {results['Area_Slice'].values[0]:.2f} mm²")
print(f"Vessel density: {results['Vessel_Length_Density'].values[0]:.3f} mm/mm²")
print(f"Branch points: {results['Number_BranchingPoints'].values[0]:.0f}")
```

## Input Image Requirements

The input image must contain the following color-coded elements:

1. **Reference line (Yellow)**: RGB ≈ (254, 251, 66)
   - Used to calibrate pixel-to-mm conversion
   - Should form a closed circle or known perimeter
   
2. **Outline (Red)**: RGB ≈ (225, 37, 1)
   - Defines the slice boundary
   - Should be a closed contour
   
3. **Vascular network (Blue)**: RGB ≈ (2, 66, 168)
   - The actual vascular structure to analyze
   - Should be clearly visible against background

## Extracted Traits

The tool extracts 16 phenotypic traits:

| # | Trait Name | Description | Units |
|---|------------|-------------|-------|
| 1 | Area_Slice | Total slice area | mm² |
| 2 | Vessel_Length_Density | Total vessel length per area | mm/mm² |
| 3 | Number_Regions | Number of closed regions | count |
| 4 | Area_mean_Regions | Mean area of regions | mm² |
| 5 | ExtendedRegions_Area_mean | Mean area of extended regions | mm² |
| 6 | Length_total_skeleton | Total skeleton length | mm |
| 7 | Number_BranchingPoints | Number of branching points | count |
| 8 | Number_InnerEndPoints | Number of inner endpoints | count |
| 9 | Branching_Point_Density_Length | Branching points per length | count/mm |
| 10 | VDD_mean | Mean vascular diffusion distance | mm |
| 11 | VDD_max | Maximum vascular diffusion distance | mm |
| 12 | Solidity_mean_Regions | Mean region solidity | ratio |
| 13 | Loopness_Regions | Regions per area (loopiness) | count/mm² |
| 14 | Hausdorf_dimension_skeleton | Fractal dimension | dimensionless |
| 15 | Eccentricity_mean_Regions | Mean region eccentricity | ratio |

### Trait Descriptions

- **Area_Slice**: Total area enclosed by the red outline
- **Vessel_Length_Density**: Total skeleton length divided by slice area
- **Number_Regions**: Closed regions formed by the vascular network
- **VDD (Vascular Diffusion Distance)**: Distance from any point to nearest vessel
- **Hausdorff Dimension**: Fractal dimension measuring network complexity
- **Solidity**: Ratio of region area to convex hull area
- **Eccentricity**: Ratio of focal distance to major axis length

## Configuration Parameters

Adjust these parameters when creating a `VascularAnalyzer`:

```python
analyzer = VascularAnalyzer(
    border_width_mm=0.75,           # Width of border region for endpoint classification
    shrinking_distance_mm=0.5,      # Distance to shrink slice edges (if do_shrink=True)
    branch_merge_distance=20,       # Merge branch points within this pixel distance
    do_shrink=False,                # Whether to apply edge shrinking
    save_debug_image=False,         # Whether to save skeleton visualization
    quiet=True                      # Suppress all output
)
```

## Edge Shrinking

The optional edge shrinking feature removes a border region from the slice:

- **Purpose**: Removes edge artifacts and vessels cut by the slice boundary
- **Effect**: Reduces slice area and removes peripheral vessels
- **When to use**: When edge effects are problematic
- **When to skip**: For maximum data retention

Enable with `do_shrink=True` or `--shrink` flag.

## Debug Visualization

Enable debug visualization to inspect the skeleton extraction:

- **Green lines**: Detected skeleton
- **Red circles**: Branch points
- **Blue circles**: Endpoints

Enable with `save_debug_image=True` or `--debug` flag.

## Output Format

Results are saved as a CSV file with one row containing all 15 traits:

```csv
Area_Slice,Vessel_Length_Density,Number_Regions,Area_mean_Regions,...
69.79,1.43,32,1.38,...
```

Load results in Python:
```python
import pandas as pd
df = pd.read_csv('results.csv')
```


## Troubleshooting

### Common Issues

**ImportError: No module named 'cv2'**
```bash
pip install opencv-python
```

**ImportError: No module named 'skimage'**
```bash
pip install scikit-image
```

**FileNotFoundError: Image not found**
- Check file path is correct
- Use absolute paths if relative paths fail
- Verify file extension (.png, .jpg, etc.)

**Warning: No reference line found**
- Verify yellow reference line is present in image
- Check RGB values match expected ranges (R>246, G>246, B<70)
- Ensure reference line forms a closed shape

**Poor or unexpected results**
- Ensure image has proper color-coded elements
- Check that vascular network is clearly visible (blue)
- Verify outline (red) forms a closed contour
- Try with/without edge shrinking (`--shrink`)
- Use `--debug` to visualize skeleton extraction

**ValueError: Could not load image**
- Verify image file is not corrupted
- Check file format is supported (PNG, JPG, TIFF)
- Ensure file has read permissions

## Technical Details

### Algorithm Overview

1. **Calibration**: Extract reference line perimeter to calculate mm/pixel conversion
2. **Preprocessing**: Optional edge shrinking to remove border artifacts
3. **Segmentation**: Extract outline and vascular network using color thresholds
4. **Skeletonization**: Convert vascular network to 1-pixel-wide skeleton
5. **Branch point merging**: Cluster nearby branch points within 20 pixels
6. **Region analysis**: Identify and measure closed regions
7. **Trait calculation**: Compute all 16 phenotypic traits
8. **Output**: Save results to CSV

### Libraries Used

- **NumPy**: Array operations and numerical computing
- **OpenCV**: Image loading and manipulation
- **SciPy**: Distance transforms, morphological operations
- **scikit-image**: Skeletonization, region properties, morphology
- **Pandas**: DataFrame creation and CSV export

## License

© Michael Henke (mhenke@gwdg.de)
© Zhanwu Dai (zhanwu.dai@ibcas.ac.cn）

This project is licensed under the MIT License.


## Citation

If using this tool in research, please cite: 
“Tackling berry vascular topology diversity across Vitis spp. via high-throughput deep learning analysis.” (In submission)



## Support

For issues or questions:
- Provide Python version (`python --version`)
- Include library versions (`pip freeze`)
- Share error message/traceback
- Sample image if possible

## Version History

- **v1.0** (2025-12-28): Initial release
  - 15 trait extraction pipeline
  - Edge shrinking option
  - Debug visualization
  - Silent mode by default
