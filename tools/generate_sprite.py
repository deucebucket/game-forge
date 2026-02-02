#!/usr/bin/env python3
"""
2D Sprite Generator - Uses Gemini Browser API + Background Removal
Generates game-ready sprites with transparency
"""

import argparse
import requests
import json
import os
import sys
import time
from pathlib import Path

# Paths
ASSETS_DIR = Path("/mnt/4tb-storage/game-forge/assets/sprites")
GEMINI_API = "http://localhost:8891"

def generate_with_gemini(prompt: str, filename: str) -> str:
    """Generate image via Gemini browser API."""
    # Enhance prompt for game sprites
    enhanced_prompt = f"Generate a game sprite: {prompt}. Make it suitable for a video game with a clean, simple background that can be easily removed."

    response = requests.post(
        f"{GEMINI_API}/gemini/prompt",
        json={"prompt": enhanced_prompt},
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.text}")

    result = response.json()
    if not result.get("success"):
        raise Exception(f"Generation failed: {result.get('error')}")

    return result.get("screenshot")

def remove_background(input_path: str, output_path: str) -> str:
    """Remove background using rembg."""
    try:
        from rembg import remove
        from PIL import Image

        with open(input_path, 'rb') as f:
            input_data = f.read()

        output_data = remove(input_data)

        # Save with transparency
        img = Image.open(io.BytesIO(output_data))
        img.save(output_path, 'PNG')

        return output_path
    except ImportError:
        print("Warning: rembg not installed. Run: pip install rembg[gpu]")
        # Fallback: just copy the file
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

def crop_to_content(image_path: str) -> str:
    """Crop image to non-transparent content."""
    try:
        from PIL import Image

        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Get bounding box of non-transparent pixels
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            img.save(image_path, 'PNG')

        return image_path
    except Exception as e:
        print(f"Warning: Could not crop image: {e}")
        return image_path

def main():
    parser = argparse.ArgumentParser(description="Generate game sprites with transparency")
    parser.add_argument("--prompt", "-p", required=True, help="Description of the sprite")
    parser.add_argument("--filename", "-f", help="Output filename (without extension)")
    parser.add_argument("--no-transparency", action="store_true", help="Skip background removal")
    parser.add_argument("--no-crop", action="store_true", help="Skip auto-cropping")

    args = parser.parse_args()

    # Generate filename if not provided
    if not args.filename:
        timestamp = int(time.time())
        safe_name = args.prompt[:30].replace(" ", "-").replace("/", "-")
        args.filename = f"sprite-{safe_name}-{timestamp}"

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating sprite: {args.prompt}")

    # Step 1: Generate with Gemini
    print("  [1/3] Generating with Gemini (Imagen 3)...")
    screenshot_path = generate_with_gemini(args.prompt, args.filename)
    print(f"  -> Screenshot: {screenshot_path}")

    # Step 2: Remove background
    output_path = ASSETS_DIR / f"{args.filename}.png"

    if not args.no_transparency:
        print("  [2/3] Removing background...")
        temp_path = ASSETS_DIR / f"{args.filename}_raw.png"

        # Copy screenshot to temp location
        import shutil
        shutil.copy(screenshot_path, temp_path)

        remove_background(str(temp_path), str(output_path))
        temp_path.unlink()  # Clean up
    else:
        print("  [2/3] Skipping background removal...")
        import shutil
        shutil.copy(screenshot_path, output_path)

    # Step 3: Crop to content
    if not args.no_crop:
        print("  [3/3] Cropping to content...")
        crop_to_content(str(output_path))
    else:
        print("  [3/3] Skipping crop...")

    print(f"\nSprite saved to: {output_path}")
    print(f"MEDIA: {output_path}")

if __name__ == "__main__":
    import io
    main()
