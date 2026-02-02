#!/usr/bin/env python3
"""
Game Asset Generator - Unified tool for 2D sprites and 3D models
Combines Gemini (2D) + rembg (AI transparency) + Meshy (3D)
"""

import argparse
import requests
import json
import os
import sys
import time
import io
from pathlib import Path

# Add the tools directory to path
TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))

# Paths
BASE_DIR = Path("/mnt/4tb-storage/game-forge")
SPRITES_DIR = BASE_DIR / "assets/sprites"
MODELS_DIR = BASE_DIR / "assets/models"
SECRETS_DIR = BASE_DIR / "secrets"
GEMINI_API = "http://localhost:8891"
MESHY_API = "https://api.meshy.ai/openapi/v2"

def get_meshy_key():
    """Load Meshy API key."""
    key_file = SECRETS_DIR / "meshy.key"
    if key_file.exists():
        return key_file.read_text().strip()
    return os.environ.get("MESHY_API_KEY")

# ============== 2D SPRITE GENERATION ==============

def generate_sprite_gemini(prompt: str) -> str:
    """Generate 2D image via Gemini browser API."""
    enhanced = f"Generate a game sprite with transparent/checkered background: {prompt}. Clean edges, suitable for video game, isolated on transparent background."

    response = requests.post(
        f"{GEMINI_API}/gemini/prompt",
        json={"prompt": enhanced},
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.text}")

    result = response.json()
    if not result.get("success"):
        raise Exception(f"Generation failed: {result.get('error')}")

    return result.get("screenshot")

def ai_remove_background(image_path: str, output_path: str) -> str:
    """Remove background using rembg AI."""
    from rembg import remove
    from PIL import Image

    with open(image_path, 'rb') as f:
        input_data = f.read()

    # rembg with optimized settings for game sprites
    output_data = remove(
        input_data,
        alpha_matting=True,  # Better edge handling
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )

    # Save with proper PNG transparency
    img = Image.open(io.BytesIO(output_data))

    # Ensure RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    img.save(output_path, 'PNG')
    return output_path

def crop_to_content(image_path: str, padding: int = 4) -> None:
    """Crop image to non-transparent content."""
    from PIL import Image

    img = Image.open(image_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    bbox = img.getbbox()
    if bbox:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        img = img.crop((x1, y1, x2, y2))
        img.save(image_path, 'PNG')

def generate_sprite(prompt: str, filename: str = None, crop: bool = True) -> str:
    """Full pipeline: Gemini generation -> AI background removal -> crop."""
    SPRITES_DIR.mkdir(parents=True, exist_ok=True)

    if not filename:
        timestamp = int(time.time())
        safe_name = prompt[:25].replace(" ", "-").replace("/", "-").lower()
        filename = f"{safe_name}-{timestamp}"

    print(f"[SPRITE] Generating: {prompt}")

    # Step 1: Generate with Gemini
    print("  [1/3] Generating with Gemini (Imagen 3)...")
    screenshot_path = generate_sprite_gemini(prompt)
    print(f"        Screenshot: {screenshot_path}")

    # Step 2: AI Background Removal
    print("  [2/3] AI background removal (rembg)...")
    output_path = str(SPRITES_DIR / f"{filename}.png")
    ai_remove_background(screenshot_path, output_path)

    # Step 3: Crop to content
    if crop:
        print("  [3/3] Cropping to content...")
        crop_to_content(output_path)
    else:
        print("  [3/3] Skipping crop")

    from PIL import Image
    img = Image.open(output_path)
    print(f"\n  Sprite saved: {output_path}")
    print(f"  Size: {img.width}x{img.height}")

    return output_path

# ============== 3D MODEL GENERATION ==============

def create_meshy_task(prompt: str, style: str = "low-poly") -> str:
    """Start a Meshy text-to-3D task."""
    api_key = get_meshy_key()
    if not api_key:
        raise Exception("Meshy API key not found")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "mode": "preview",
        "prompt": prompt,
        "art_style": style,
        "negative_prompt": "low quality, blurry, distorted, ugly"
    }

    response = requests.post(
        f"{MESHY_API}/text-to-3d",
        headers=headers,
        json=data,
        timeout=30
    )

    if response.status_code not in [200, 202]:
        raise Exception(f"Meshy API error: {response.status_code} - {response.text}")

    result = response.json()
    return result.get("result")

def wait_meshy_task(task_id: str, max_wait: int = 300) -> dict:
    """Wait for Meshy task completion."""
    api_key = get_meshy_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    last_progress = -1

    while time.time() - start < max_wait:
        response = requests.get(
            f"{MESHY_API}/text-to-3d/{task_id}",
            headers=headers,
            timeout=30
        )
        status = response.json()
        state = status.get("status")
        progress = status.get("progress", 0)

        if progress != last_progress:
            print(f"        Progress: {progress}%")
            last_progress = progress

        if state == "SUCCEEDED":
            return status
        elif state == "FAILED":
            raise Exception(f"Generation failed: {status.get('message')}")

        time.sleep(5)

    raise Exception("Timeout waiting for Meshy")

def download_file(url: str, output_path: str) -> str:
    """Download a file."""
    response = requests.get(url, timeout=120)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    return output_path

def generate_model(prompt: str, filename: str = None, style: str = "low-poly", format: str = "glb") -> str:
    """Full pipeline: Meshy 3D generation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not filename:
        timestamp = int(time.time())
        safe_name = prompt[:25].replace(" ", "-").replace("/", "-").lower()
        filename = f"{safe_name}-{timestamp}"

    print(f"[MODEL] Generating: {prompt}")
    print(f"        Style: {style}")

    # Step 1: Start generation
    print("  [1/3] Starting Meshy generation...")
    task_id = create_meshy_task(prompt, style)
    print(f"        Task ID: {task_id}")

    # Step 2: Wait for completion
    print("  [2/3] Waiting for generation (~1-3 minutes)...")
    result = wait_meshy_task(task_id)

    # Step 3: Download
    print("  [3/3] Downloading model...")
    model_urls = result.get("model_urls", {})
    model_url = model_urls.get(format) or model_urls.get("glb")

    if not model_url:
        raise Exception(f"No URL for format: {format}")

    output_path = str(MODELS_DIR / f"{filename}.{format}")
    download_file(model_url, output_path)

    # Also get thumbnail
    thumb_url = result.get("thumbnail_url")
    if thumb_url:
        thumb_path = str(MODELS_DIR / f"{filename}_preview.png")
        download_file(thumb_url, thumb_path)
        print(f"        Preview: {thumb_path}")

    print(f"\n  Model saved: {output_path}")

    return output_path

# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(
        description="Game Asset Generator - 2D Sprites & 3D Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 2D sprite
  %(prog)s sprite "a cute robot character"

  # Generate 3D model
  %(prog)s model "a treasure chest" --style low-poly

  # Quick sprite (no crop)
  %(prog)s sprite "health potion" --no-crop
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sprite command
    sprite_parser = subparsers.add_parser("sprite", help="Generate 2D sprite")
    sprite_parser.add_argument("prompt", help="Description of the sprite")
    sprite_parser.add_argument("--filename", "-f", help="Output filename (no extension)")
    sprite_parser.add_argument("--no-crop", action="store_true", help="Don't crop to content")

    # Model command
    model_parser = subparsers.add_parser("model", help="Generate 3D model")
    model_parser.add_argument("prompt", help="Description of the model")
    model_parser.add_argument("--filename", "-f", help="Output filename (no extension)")
    model_parser.add_argument("--style", "-s", default="low-poly",
                             choices=["realistic", "cartoon", "low-poly", "sculpture"],
                             help="Art style")
    model_parser.add_argument("--format", default="glb",
                             choices=["glb", "fbx", "obj"],
                             help="Output format")

    args = parser.parse_args()

    try:
        if args.command == "sprite":
            output = generate_sprite(
                args.prompt,
                filename=args.filename,
                crop=not args.no_crop
            )
        elif args.command == "model":
            output = generate_model(
                args.prompt,
                filename=args.filename,
                style=args.style,
                format=args.format
            )

        print(f"\nMEDIA: {output}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
