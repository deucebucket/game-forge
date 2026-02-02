#!/usr/bin/env python3
"""
Smart Transparency Tool for AI-Generated Images
Handles checkered backgrounds, solid backgrounds, and edge cleanup
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import colorsys

def detect_checkered_pattern(img_array: np.ndarray, sample_size: int = 50) -> bool:
    """Detect if image has a checkered transparency pattern."""
    # Sample corners where transparency usually is
    h, w = img_array.shape[:2]

    corners = [
        img_array[0:sample_size, 0:sample_size],  # top-left
        img_array[0:sample_size, w-sample_size:w],  # top-right
        img_array[h-sample_size:h, 0:sample_size],  # bottom-left
        img_array[h-sample_size:h, w-sample_size:w],  # bottom-right
    ]

    for corner in corners:
        if corner.size == 0:
            continue

        # Check for alternating pattern (typical checkered is ~8-16px squares)
        # Look for high variance in small regions
        gray = np.mean(corner, axis=2) if len(corner.shape) == 3 else corner

        # Check horizontal variance
        h_diff = np.abs(np.diff(gray, axis=1))
        # Check vertical variance
        v_diff = np.abs(np.diff(gray, axis=0))

        # Checkered patterns have regular high-frequency changes
        if np.mean(h_diff > 20) > 0.3 or np.mean(v_diff > 20) > 0.3:
            # Further verify: check if it's gray/white checkered
            unique_colors = len(np.unique(corner.reshape(-1, corner.shape[-1]), axis=0))
            if unique_colors < 10:  # Checkered usually has just 2-4 colors
                return True

    return False

def remove_checkered_background(img: Image.Image) -> Image.Image:
    """Remove checkered transparency pattern and replace with actual alpha."""
    img_array = np.array(img.convert('RGBA'))

    # Common checkered colors (light gray / white or gray / darker gray)
    # Typical Photoshop: (255,255,255) and (204,204,204)
    # Some tools use: (192,192,192) and (255,255,255)

    checkered_colors = [
        [(255, 255, 255), (204, 204, 204)],  # Photoshop style
        [(255, 255, 255), (192, 192, 192)],  # Alternative
        [(238, 238, 238), (255, 255, 255)],  # Light
        [(200, 200, 200), (230, 230, 230)],  # Medium
        [(170, 170, 170), (200, 200, 200)],  # Darker
    ]

    # Create mask for checkered regions
    h, w = img_array.shape[:2]
    alpha_mask = np.ones((h, w), dtype=np.uint8) * 255

    # For each pixel, check if it matches checkered pattern colors
    for colors in checkered_colors:
        for color in colors:
            # Allow some tolerance for color matching
            tolerance = 15
            mask = (
                (np.abs(img_array[:,:,0].astype(int) - color[0]) < tolerance) &
                (np.abs(img_array[:,:,1].astype(int) - color[1]) < tolerance) &
                (np.abs(img_array[:,:,2].astype(int) - color[2]) < tolerance)
            )
            alpha_mask[mask] = 0

    # Apply alpha
    img_array[:,:,3] = alpha_mask

    return Image.fromarray(img_array)

def remove_solid_background(img: Image.Image, color: tuple = None, tolerance: int = 30) -> Image.Image:
    """Remove solid color background."""
    img_array = np.array(img.convert('RGBA'))

    if color is None:
        # Auto-detect: sample corners for most common color
        h, w = img_array.shape[:2]
        corners = np.concatenate([
            img_array[0:10, 0:10].reshape(-1, 4),
            img_array[0:10, w-10:w].reshape(-1, 4),
            img_array[h-10:h, 0:10].reshape(-1, 4),
            img_array[h-10:h, w-10:w].reshape(-1, 4),
        ])

        # Find most common color
        unique, counts = np.unique(corners[:,:3], axis=0, return_counts=True)
        color = tuple(unique[np.argmax(counts)])
        print(f"  Auto-detected background color: RGB{color}")

    # Create mask
    mask = (
        (np.abs(img_array[:,:,0].astype(int) - color[0]) < tolerance) &
        (np.abs(img_array[:,:,1].astype(int) - color[1]) < tolerance) &
        (np.abs(img_array[:,:,2].astype(int) - color[2]) < tolerance)
    )

    img_array[:,:,3][mask] = 0

    return Image.fromarray(img_array)

def cleanup_edges(img: Image.Image, iterations: int = 1) -> Image.Image:
    """Clean up jagged edges and semi-transparent pixels."""
    from PIL import ImageFilter

    img_array = np.array(img.convert('RGBA'))

    # Slight blur on alpha channel to smooth edges
    alpha = Image.fromarray(img_array[:,:,3])
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.5))
    img_array[:,:,3] = np.array(alpha)

    # Remove very low alpha pixels (noise)
    img_array[:,:,3][img_array[:,:,3] < 20] = 0

    # Make near-opaque pixels fully opaque
    img_array[:,:,3][img_array[:,:,3] > 235] = 255

    return Image.fromarray(img_array)

def crop_to_content(img: Image.Image, padding: int = 2) -> Image.Image:
    """Crop to non-transparent content with optional padding."""
    bbox = img.getbbox()
    if bbox:
        # Add padding
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        return img.crop((x1, y1, x2, y2))
    return img

def main():
    parser = argparse.ArgumentParser(description="Make images transparent for game use")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output", "-o", help="Output path (default: input_transparent.png)")
    parser.add_argument("--method", "-m", default="auto",
                       choices=["auto", "checkered", "solid", "rembg"],
                       help="Transparency method")
    parser.add_argument("--color", "-c", help="Background color to remove (R,G,B)")
    parser.add_argument("--tolerance", "-t", type=int, default=30,
                       help="Color matching tolerance (default: 30)")
    parser.add_argument("--no-crop", action="store_true", help="Don't crop to content")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up edges")
    parser.add_argument("--padding", "-p", type=int, default=2, help="Crop padding")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = args.output or input_path.with_stem(input_path.stem + "_transparent")

    print(f"Processing: {input_path}")

    # Load image
    img = Image.open(input_path).convert('RGBA')
    img_array = np.array(img)

    # Determine method
    method = args.method
    if method == "auto":
        if detect_checkered_pattern(img_array):
            method = "checkered"
            print("  Detected: Checkered pattern background")
        else:
            method = "solid"
            print("  Detected: Solid color background")

    # Apply transparency
    if method == "checkered":
        print("  Removing checkered background...")
        img = remove_checkered_background(img)
    elif method == "solid":
        color = None
        if args.color:
            color = tuple(map(int, args.color.split(',')))
        print("  Removing solid background...")
        img = remove_solid_background(img, color, args.tolerance)
    elif method == "rembg":
        print("  Using rembg AI background removal...")
        try:
            from rembg import remove
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            result = remove(img_bytes.getvalue())
            img = Image.open(io.BytesIO(result))
        except ImportError:
            print("  Error: rembg not installed. Run: pip install rembg[gpu]")
            sys.exit(1)

    # Cleanup edges
    if not args.no_cleanup:
        print("  Cleaning up edges...")
        img = cleanup_edges(img)

    # Crop to content
    if not args.no_crop:
        print("  Cropping to content...")
        img = crop_to_content(img, args.padding)

    # Save
    img.save(output_path, 'PNG')
    print(f"\nSaved: {output_path}")
    print(f"Size: {img.width}x{img.height}")
    print(f"MEDIA: {output_path}")

if __name__ == "__main__":
    import sys
    main()
