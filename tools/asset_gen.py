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

def create_meshy_image_to_3d(image_path: str, prompt: str = None) -> str:
    """Start a Meshy image-to-3D task using a reference image."""
    import base64

    api_key = get_meshy_key()
    if not api_key:
        raise Exception("Meshy API key not found")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Read and encode image as base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Determine mime type
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    data_uri = f"data:{mime};base64,{image_data}"

    data = {
        "image_url": data_uri,
        "ai_model": "meshy-5",  # Latest stable model
        "enable_pbr": True,  # Generate PBR textures
        "topology": "triangle",
        "target_polycount": 30000  # Good for games
    }

    response = requests.post(
        f"{MESHY_API}/image-to-3d",
        headers=headers,
        json=data,
        timeout=60
    )

    if response.status_code not in [200, 202]:
        raise Exception(f"Meshy API error: {response.status_code} - {response.text}")

    result = response.json()
    return result.get("result")

def wait_meshy_image_task(task_id: str, max_wait: int = 600) -> dict:
    """Wait for Meshy image-to-3D task completion (takes longer than text-to-3D)."""
    api_key = get_meshy_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    last_progress = -1

    while time.time() - start < max_wait:
        response = requests.get(
            f"{MESHY_API}/image-to-3d/{task_id}",
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
            raise Exception(f"Generation failed: {status.get('task_error', {}).get('message', 'Unknown error')}")

        time.sleep(10)

    raise Exception("Timeout waiting for Meshy image-to-3D")

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

def generate_reference_image(prompt: str, filename: str) -> str:
    """Generate a clean reference image for 3D conversion via Gemini."""
    SPRITES_DIR.mkdir(parents=True, exist_ok=True)

    # Optimize prompt for 3D reference
    enhanced = f"Create a reference image for 3D modeling: {prompt}. Front-facing view, centered, clean white or light gray background, full body visible, good lighting, no text or watermarks."

    response = requests.post(
        f"{GEMINI_API}/gemini/prompt",
        json={"prompt": enhanced},
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.text}")

    result = response.json()
    screenshot_path = result.get("screenshot")

    # AI background removal for cleaner reference
    from rembg import remove
    from PIL import Image

    with open(screenshot_path, 'rb') as f:
        input_data = f.read()

    output_data = remove(input_data)
    img = Image.open(io.BytesIO(output_data))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Save reference image
    ref_path = str(SPRITES_DIR / f"{filename}_reference.png")
    img.save(ref_path, 'PNG')

    return ref_path

def generate_model(prompt: str, filename: str = None, style: str = "low-poly", format: str = "glb", use_reference: bool = True) -> str:
    """Full pipeline: Generate reference with Gemini, then Meshy image-to-3D."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not filename:
        timestamp = int(time.time())
        safe_name = prompt[:25].replace(" ", "-").replace("/", "-").lower()
        filename = f"{safe_name}-{timestamp}"

    print(f"[MODEL] Generating: {prompt}")

    if use_reference:
        # Smart pipeline: Gemini reference -> Meshy image-to-3D
        print("  [1/4] Generating reference image with Gemini...")
        ref_path = generate_reference_image(prompt, filename)
        print(f"        Reference: {ref_path}")

        print("  [2/4] Starting Meshy image-to-3D...")
        task_id = create_meshy_image_to_3d(ref_path, prompt)
        print(f"        Task ID: {task_id}")

        print("  [3/4] Waiting for 3D generation (~3-5 minutes)...")
        result = wait_meshy_image_task(task_id)

        step = "[4/4]"
    else:
        # Fallback: Text-to-3D only
        print(f"        Style: {style}")
        print("  [1/3] Starting Meshy text-to-3D...")
        task_id = create_meshy_task(prompt, style)
        print(f"        Task ID: {task_id}")

        print("  [2/3] Waiting for generation (~1-3 minutes)...")
        result = wait_meshy_task(task_id)

        step = "[3/3]"

    # Download model
    print(f"  {step} Downloading model...")
    model_urls = result.get("model_urls", {})
    model_url = model_urls.get(format) or model_urls.get("glb")

    if not model_url:
        raise Exception(f"No URL for format: {format}")

    output_path = str(MODELS_DIR / f"{filename}.{format}")
    download_file(model_url, output_path)

    # Download thumbnail
    thumb_url = result.get("thumbnail_url")
    if thumb_url:
        thumb_path = str(MODELS_DIR / f"{filename}_preview.png")
        download_file(thumb_url, thumb_path)
        print(f"        Preview: {thumb_path}")

    # Download PBR textures if available
    texture_urls = result.get("texture_urls", [])
    if texture_urls:
        for i, tex_url in enumerate(texture_urls):
            tex_path = str(MODELS_DIR / f"{filename}_texture_{i}.png")
            download_file(tex_url, tex_path)
        print(f"        Textures: {len(texture_urls)} downloaded")

    print(f"\n  Model saved: {output_path}")

    return output_path

# ============== TEXTURE GENERATION ==============

def generate_texture(prompt: str, filename: str = None, tex_type: str = "diffuse",
                    size: str = "512", tileable: bool = False) -> str:
    """Generate game textures via Gemini."""
    SPRITES_DIR.mkdir(parents=True, exist_ok=True)

    if not filename:
        timestamp = int(time.time())
        safe_name = prompt[:20].replace(" ", "-").replace("/", "-").lower()
        filename = f"texture-{safe_name}-{timestamp}"

    # Build texture-specific prompt
    tile_hint = "seamless tileable pattern, edges match perfectly," if tileable else ""
    type_hints = {
        "diffuse": "color texture, base color map, albedo,",
        "normal": "normal map, blue-purple tinted, bump detail, surface normals,",
        "roughness": "roughness map, grayscale, white=rough black=smooth,",
        "all": "PBR texture set preview showing diffuse normal and roughness,"
    }

    enhanced = f"Generate a {size}x{size} game texture: {type_hints.get(tex_type, '')} {tile_hint} {prompt}. Top-down view, flat, no perspective, suitable for 3D model UV mapping."

    print(f"[TEXTURE] Generating: {prompt}")
    print(f"          Type: {tex_type}, Size: {size}x{size}, Tileable: {tileable}")

    # Generate with Gemini
    print("  [1/2] Generating with Gemini...")
    response = requests.post(
        f"{GEMINI_API}/gemini/prompt",
        json={"prompt": enhanced},
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.text}")

    result = response.json()
    screenshot_path = result.get("screenshot")

    # Process: crop to texture area and resize
    print("  [2/2] Processing texture...")
    from PIL import Image

    img = Image.open(screenshot_path)

    # Try to extract just the texture from the screenshot
    # (Gemini screenshots include browser chrome)
    # Crop to center region where texture likely is
    w, h = img.size
    # Assume texture is roughly centered
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.15)
    img = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))

    # Resize to requested size
    target_size = int(size)
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Save
    output_path = str(SPRITES_DIR / f"{filename}_{tex_type}.png")
    img.save(output_path, 'PNG')

    print(f"\n  Texture saved: {output_path}")
    print(f"  Size: {target_size}x{target_size}")

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
                             help="Art style (for text-to-3D fallback)")
    model_parser.add_argument("--format", default="glb",
                             choices=["glb", "fbx", "obj"],
                             help="Output format")
    model_parser.add_argument("--no-reference", action="store_true",
                             help="Skip Gemini reference image, use text-to-3D only")

    # Texture command
    texture_parser = subparsers.add_parser("texture", help="Generate texture/material")
    texture_parser.add_argument("prompt", help="Description of the texture")
    texture_parser.add_argument("--filename", "-f", help="Output filename")
    texture_parser.add_argument("--type", "-t", default="diffuse",
                               choices=["diffuse", "normal", "roughness", "all"],
                               help="Texture type to generate")
    texture_parser.add_argument("--size", default="512",
                               choices=["256", "512", "1024", "2048"],
                               help="Texture size (square)")
    texture_parser.add_argument("--tileable", action="store_true",
                               help="Make texture tileable/seamless")

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
                format=args.format,
                use_reference=not args.no_reference
            )
        elif args.command == "texture":
            output = generate_texture(
                args.prompt,
                filename=args.filename,
                tex_type=args.type,
                size=args.size,
                tileable=args.tileable
            )

        print(f"\nMEDIA: {output}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
