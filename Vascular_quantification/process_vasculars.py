#!/usr/bin/env python3
"""
Vascular Trait Extraction - Python Implementation

Extracts phenotypic traits from vascular networks in fruit slice images.
Measures 16 different characteristics including area, vessel density,
branching points, and fractal dimensions.

Author: Michael Henke, 20251229, IB-CAS
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist, squareform
from skimage import morphology, measure
from skimage.measure import label, regionprops
from skimage.morphology import (
    skeletonize, remove_small_objects, binary_dilation,
    binary_closing, binary_opening, disk, square
)
import pandas as pd
from pathlib import Path


class VascularAnalyzer:
    """Analyzes vascular networks in fruit slice images."""

    def __init__(self,
                 border_width_mm=0.75,
                 shrinking_distance_mm=0.5,
                 branch_merge_distance=20,
                 do_shrink=False,
                 save_debug_image=False,
                 quiet=True):
        """
        Initialize analyzer with configuration.

        Args:
            border_width_mm: Width of border region for endpoint classification
            shrinking_distance_mm: Distance to shrink slice edges
            branch_merge_distance: Merge branch points within this pixel distance
            do_shrink: Whether to apply edge shrinking preprocessing
            save_debug_image: Whether to save skeleton visualization
            quiet: If True, suppress all output (default: True)
        """
        self.border_width_mm = border_width_mm
        self.shrinking_distance_mm = shrinking_distance_mm
        self.branch_merge_distance = branch_merge_distance
        self.do_shrink = do_shrink
        self.save_debug_image = save_debug_image
        self.quiet = quiet

    @staticmethod
    def _remove_small_objects_safe(binary_img, min_size):
        """Remove small objects without warnings."""
        labeled = measure.label(binary_img)
        if labeled.max() > 1:
            return remove_small_objects(labeled, min_size=min_size) > 0
        return labeled > 0

    def process(self, image_path, output_csv=None):
        """
        Extract vascular traits from image.

        Args:
            image_path: Path to input image
            output_csv: Optional path for CSV output

        Returns:
            DataFrame with extracted traits
        """
        # Load and validate image
        img = self._load_image(image_path)

        # Extract calibration from reference line
        self.pix2mm = self._calculate_calibration(img)

        # Preprocess image
        if self.do_shrink:
            img = self._shrink_edges(img)

        # Crop to content
        img = self._crop_to_content(img)

        # Extract components
        outline = self._extract_outline(img)
        outline_filled = ndimage.binary_fill_holes(outline)
        vascular = self._extract_vascular(img)

        # Calculate border area for endpoint classification
        border_area = self._calculate_border_area(outline_filled)

        # Create skeleton
        skeleton = self._create_skeleton(vascular)

        # Analyze skeleton components
        branch_points, end_points = self._find_critical_points(skeleton)
        branch_points = self._merge_branch_points(branch_points, self.branch_merge_distance)
        inner_endpoints = self._classify_endpoints(end_points, border_area)

        # Save debug visualization if requested
        if self.save_debug_image:
            debug_path = Path(image_path).stem + '_skeleton_debug.png'
            self._save_skeleton_debug(img, skeleton, branch_points, end_points, debug_path)

        # Calculate regions
        regions = self._analyze_regions(outline_filled, skeleton)

        # Extended skeleton and regions
        extended_skel = self._extend_skeleton(skeleton, end_points, inner_endpoints, outline)
        extended_regions = self._analyze_extended_regions(outline_filled, extended_skel)

        # VDD map
        vdd_map = self._calculate_vdd(extended_skel, outline_filled)

        # Segment lengths
        segment_lengths = self._calculate_segment_lengths(skeleton, branch_points)

        # Fractal dimension
        hausdorff_dim = self._calculate_hausdorff_dimension(skeleton)

        # Compile traits
        traits = self._compile_traits(
            outline_filled, skeleton, regions, extended_regions,
            branch_points, end_points, inner_endpoints,
            segment_lengths, vdd_map, hausdorff_dim
        )

        # Save results
        df = pd.DataFrame([traits])
        if output_csv is None:
            output_csv = Path(image_path).with_suffix('.csv')
        df.to_csv(output_csv, index=False)

        return df

    def _load_image(self, path):
        """Load and convert image to RGB."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _calculate_calibration(self, img):
        """Calculate pixel to mm conversion from yellow reference line."""
        refline = (img[:,:,0] > 246) & (img[:,:,1] > 246) & (img[:,:,2] < 70)

        if not np.any(refline):
            return 0.005  # Default ~200 pixels per mm

        props = measure.regionprops(measure.label(refline))
        if len(props) == 0:
            return 0.005

        perimeter = props[0].perimeter
        mm = perimeter / 2
        return 1.0 / mm

    def _shrink_edges(self, img):
        """Shrink slice edges to remove border artifacts."""
        outline = (img[:,:,0] > 205) & (img[:,:,1] < 58) & (img[:,:,2] < 26)
        outline = binary_closing(outline, disk(25))
        outline = self._remove_small_objects_safe(outline, 1000)

        outline_filled = ndimage.binary_fill_holes(outline)

        dist_from_edge = ndimage.distance_transform_edt(outline_filled)
        shrink_pixels = self.shrinking_distance_mm / self.pix2mm
        border_mask = (dist_from_edge > 0) & (dist_from_edge < shrink_pixels)

        vascular_mask = (img[:,:,0] < 65) & (img[:,:,1] < 100) & (img[:,:,2] > 140)
        vascular_in_border = vascular_mask & border_mask

        img_copy = img.copy()
        img_copy[border_mask] = 0
        img_copy[vascular_in_border] = 0

        new_outline = outline_filled & ~border_mask
        new_outline = morphology.binary_erosion(new_outline, morphology.disk(2))
        boundary = morphology.binary_dilation(new_outline, morphology.disk(2)) & ~new_outline
        img_copy[boundary, :] = [225, 37, 1]

        return img_copy

    def _crop_to_content(self, img):
        """Crop image to bounding box of content."""
        outline = (img[:,:,0] > 180) & (img[:,:,1] < 80) & (img[:,:,2] < 40)
        outline = binary_closing(outline, disk(25))
        outline = self._remove_small_objects_safe(outline, 1000)

        if not np.any(outline):
            return img

        props = measure.regionprops(measure.label(outline))
        if len(props) == 0:
            return img

        bbox = props[0].bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]

        if height < img.shape[0] * 0.3 or width < img.shape[1] * 0.3:
            return img

        return img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    def _extract_outline(self, img):
        """Extract red outline and return filled region."""
        outline = (img[:,:,0] > 150) & (img[:,:,1] < 100) & (img[:,:,2] < 80)
        outline = binary_closing(outline, disk(50))
        outline = self._remove_small_objects_safe(outline, 100)

        filled = ndimage.binary_fill_holes(outline)

        if np.sum(filled) == np.sum(outline):
            coords = np.column_stack(np.where(outline))
            if len(coords) > 0:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(coords)
                hull_points = coords[hull.vertices]
                filled = np.zeros(outline.shape, dtype=np.uint8)
                hull_points_cv = hull_points[:, [1, 0]].astype(np.int32)
                cv2.fillPoly(filled, [hull_points_cv], 1)
                filled = filled.astype(bool)

        return filled

    def _extract_vascular(self, img):
        """Extract blue vascular network."""
        vascular = (img[:,:,0] < 65) & (img[:,:,1] < 100) & (img[:,:,2] > 140)
        return self._remove_small_objects_safe(vascular, 100)

    def _calculate_border_area(self, outline_filled):
        """Calculate border region for endpoint classification."""
        dist_from_edge = ndimage.distance_transform_edt(outline_filled)
        border_pixels = self.border_width_mm / self.pix2mm
        return (dist_from_edge > 0) & (dist_from_edge < border_pixels)

    def _create_skeleton(self, vascular):
        """Create skeleton."""
        skel = skeletonize(vascular)
        skel = morphology.thin(skel, max_num_iter=10)
        return skel

    def _find_critical_points(self, skel):
        """Find branch points and endpoints."""
        branch_points = np.zeros_like(skel, dtype=bool)
        end_points = np.zeros_like(skel, dtype=bool)

        for i in range(1, skel.shape[0]-1):
            for j in range(1, skel.shape[1]-1):
                if skel[i,j]:
                    neighbors = np.sum(skel[i-1:i+2, j-1:j+2]) - 1
                    if neighbors >= 3:
                        branch_points[i,j] = True
                    elif neighbors == 1:
                        end_points[i,j] = True

        return branch_points, end_points

    def _merge_branch_points(self, branch_points, min_distance=20):
        """Merge branch points within min_distance pixels."""
        coords = np.column_stack(np.where(branch_points))

        if len(coords) == 0:
            return branch_points

        D = squareform(pdist(coords))

        merged_coords = []
        used = set()

        for i in range(len(coords)):
            if i in used:
                continue

            cluster_indices = [i]
            close_to_i = np.where(D[i, :] <= min_distance)[0]

            for j in close_to_i:
                if j != i and j not in used:
                    cluster_indices.append(j)
                    used.add(j)

            used.add(i)

            cluster_coords = coords[cluster_indices]
            centroid = np.round(np.mean(cluster_coords, axis=0)).astype(int)
            merged_coords.append(centroid)

        merged_branch_points = np.zeros_like(branch_points, dtype=bool)
        for coord in merged_coords:
            y, x = coord
            if 0 <= y < branch_points.shape[0] and 0 <= x < branch_points.shape[1]:
                merged_branch_points[y, x] = True

        return merged_branch_points

    def _classify_endpoints(self, endpoints, border_area):
        """Classify endpoints as inner (not in border) or outer (in border)."""
        coords = np.column_stack(np.where(endpoints))
        is_outer = np.array([border_area[c[0], c[1]] for c in coords])
        return ~is_outer

    def _analyze_regions(self, outline_filled, skeleton):
        """Analyze regions created by vascular network."""
        skel_dilated = binary_dilation(skeleton, disk(1))
        regions = outline_filled & ~skel_dilated
        regions_labeled = measure.label(regions)

        props = measure.regionprops(regions_labeled)
        if len(props) > 0:
            props = sorted(props, key=lambda p: p.area, reverse=True)
            if len(props) > 1:
                return props[1:]
        return props

    def _extend_skeleton(self, skeleton, endpoints, is_inner, outline):
        """Extend outer endpoints to outline."""
        extended = skeleton.copy()

        coords = np.column_stack(np.where(endpoints))
        outer_coords = coords[~is_inner]

        if len(outer_coords) == 0 or not np.any(outline):
            return extended

        outline_coords = np.column_stack(np.where(morphology.binary_erosion(outline)))

        for coord in outer_coords:
            distances = cdist([coord], outline_coords)
            nearest = outline_coords[np.argmin(distances)]
            extended = self._draw_line(extended, coord, nearest)

        return extended

    def _draw_line(self, img, start, end):
        """Draw line between two points on binary image."""
        y1, x1 = start
        y2, x2 = end

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        while True:
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = True

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return img

    def _analyze_extended_regions(self, outline_filled, extended_skel):
        """Analyze regions with extended skeleton."""
        skel_dilated = binary_dilation(extended_skel, disk(1))
        regions = outline_filled & ~skel_dilated
        regions_labeled = measure.label(regions)

        props = measure.regionprops(regions_labeled)
        return [p for p in props if p.area >= 1000]

    def _calculate_vdd(self, skeleton, outline_filled):
        """Calculate Vascular Diffusion Distance map."""
        vdd = ndimage.distance_transform_edt(~skeleton)
        vdd[~outline_filled] = np.nan
        return vdd * self.pix2mm

    def _calculate_segment_lengths(self, skeleton, branch_points):
        """Calculate lengths of skeleton segments."""
        branch_dilated = binary_dilation(branch_points, disk(1))
        segments = skeleton & ~branch_dilated
        segments_labeled = measure.label(segments)

        props = measure.regionprops(segments_labeled)
        lengths = [p.area * self.pix2mm for p in props if p.area >= 10]

        return np.array(lengths)

    def _calculate_hausdorff_dimension(self, skeleton):
        """Calculate Hausdorff fractal dimension using box counting."""
        if np.sum(skeleton) == 0:
            return 0.0

        max_dim = max(skeleton.shape)
        size = 2 ** int(np.ceil(np.log2(max_dim)))
        padded = np.pad(skeleton,
                       ((0, size - skeleton.shape[0]),
                        (0, size - skeleton.shape[1])),
                       mode='constant')

        box_counts = []
        resolutions = []

        box_size = size
        while box_size >= 1:
            count = 0
            boxes_per_dim = size // box_size

            for i in range(boxes_per_dim):
                for j in range(boxes_per_dim):
                    box = padded[i*box_size:(i+1)*box_size,
                                j*box_size:(j+1)*box_size]
                    if np.any(box):
                        count += 1

            box_counts.append(count)
            resolutions.append(1.0 / box_size)
            box_size //= 2

        coeffs = np.polyfit(np.log(resolutions), np.log(box_counts), 1)
        return coeffs[0]

    def _save_skeleton_debug(self, img, skeleton, branch_points, end_points, filename):
        """Save skeleton with branch points and endpoints marked."""
        vis = img.copy()
        vis[skeleton] = [0, 255, 0]

        branch_coords = np.column_stack(np.where(branch_points))
        for coord in branch_coords:
            y, x = coord
            cv2.circle(vis, (x, y), 10, (255, 0, 0), -1)

        end_coords = np.column_stack(np.where(end_points))
        for coord in end_coords:
            y, x = coord
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

        cv2.imwrite(filename, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def _compile_traits(self, outline_filled, skeleton, regions, extended_regions,
                       branch_points, end_points, inner_endpoints,
                       segment_lengths, vdd_map, hausdorff_dim):
        """Compile all traits into dictionary."""
        area_slice = np.sum(outline_filled) * self.pix2mm**2
        total_length = np.sum(segment_lengths)
        n_branches = np.sum(branch_points)
        n_inner = np.sum(inner_endpoints)

        return {
            'Area_Slice': area_slice,
            'Vessel_Length_Density': total_length / area_slice if area_slice > 0 else 0,
            'Number_Regions': len(regions),
            'Area_mean_Regions': np.mean([r.area * self.pix2mm**2 for r in regions]) if regions else 0,
            'ExtendedRegions_Area_mean': np.mean([r.area * self.pix2mm**2 for r in extended_regions]) if extended_regions else 0,
            'Length_total_skeleton': total_length,
            'Number_BranchingPoints': n_branches,
            'Number_InnerEndPoints': n_inner,
            'Branching_Point_Density_Length': n_branches / total_length if total_length > 0 else 0,
            'VDD_mean': np.nanmean(vdd_map),
            'VDD_max': np.nanmax(vdd_map),
            'Solidity_mean_Regions': np.mean([r.solidity for r in regions]) if regions else 0,
            'Loopness_Regions': len(regions) / area_slice if area_slice > 0 else 0,
            'Hausdorf_dimension_skeleton': hausdorff_dim,
            'Eccentricity_mean_Regions': np.mean([r.eccentricity for r in regions]) if regions else 0,
        }


def process_vasculars(image_path, output_csv=None, **config):
    """
    Convenience function to process a single image.

    Args:
        image_path: Path to input image
        output_csv: Optional output path
        **config: Configuration parameters for VascularAnalyzer

    Returns:
        DataFrame with results
    """
    analyzer = VascularAnalyzer(**config)
    return analyzer.process(image_path, output_csv)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python process_vasculars_final.py <input_image> [output_csv]")
        print("\nOptions:")
        print("  --shrink: Enable edge shrinking")
        print("  --debug: Save skeleton debug image")
        print("  --verbose: Show processing output")
        sys.exit(1)

    image_path = sys.argv[1]
    output_csv = None
    do_shrink = '--shrink' in sys.argv
    save_debug = '--debug' in sys.argv
    quiet = '--verbose' not in sys.argv

    for arg in sys.argv[2:]:
        if not arg.startswith('--'):
            output_csv = arg

    df = process_vasculars(image_path, output_csv,
                          do_shrink=do_shrink,
                          save_debug_image=save_debug,
                          quiet=quiet)

    if not quiet:
        print("Extracted traits:")
        print(df.T)
