#!/usr/bin/env python3
"""
3D Model Generator - Uses Meshy API
Generates game-ready 3D models
"""

import argparse
import requests
import json
import os
import sys
import time
from pathlib import Path

# Paths
ASSETS_DIR = Path("/mnt/4tb-storage/game-forge/assets/models")
SECRETS_DIR = Path("/mnt/4tb-storage/game-forge/secrets")
MESHY_API = "https://api.meshy.ai/openapi/v2"

def get_api_key():
    """Load Meshy API key."""
    key_file = SECRETS_DIR / "meshy.key"
    if key_file.exists():
        return key_file.read_text().strip()
    # Fallback to environment variable
    return os.environ.get("MESHY_API_KEY")

def create_text_to_3d(prompt: str, style: str = "realistic") -> str:
    """Start a text-to-3D generation task."""
    api_key = get_api_key()
    if not api_key:
        raise Exception("Meshy API key not found")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "mode": "preview",  # preview is faster, refine for higher quality
        "prompt": prompt,
        "art_style": style,  # realistic, cartoon, low-poly, sculpture
        "negative_prompt": "low quality, blurry, distorted"
    }

    response = requests.post(
        f"{MESHY_API}/text-to-3d",
        headers=headers,
        json=data,
        timeout=30
    )

    if response.status_code != 202:
        raise Exception(f"Meshy API error: {response.status_code} - {response.text}")

    result = response.json()
    return result.get("result")  # Task ID

def check_task_status(task_id: str) -> dict:
    """Check the status of a generation task."""
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(
        f"{MESHY_API}/text-to-3d/{task_id}",
        headers=headers,
        timeout=30
    )

    return response.json()

def download_model(url: str, output_path: str) -> str:
    """Download the generated model."""
    response = requests.get(url, timeout=120)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    return output_path

def wait_for_completion(task_id: str, max_wait: int = 300) -> dict:
    """Wait for task completion with progress updates."""
    start = time.time()
    last_progress = -1

    while time.time() - start < max_wait:
        status = check_task_status(task_id)
        state = status.get("status")
        progress = status.get("progress", 0)

        if progress != last_progress:
            print(f"  Progress: {progress}%")
            last_progress = progress

        if state == "SUCCEEDED":
            return status
        elif state == "FAILED":
            raise Exception(f"Generation failed: {status.get('message')}")
        elif state in ["PENDING", "IN_PROGRESS"]:
            time.sleep(5)
        else:
            time.sleep(5)

    raise Exception("Timeout waiting for generation")

def main():
    parser = argparse.ArgumentParser(description="Generate 3D models with Meshy")
    parser.add_argument("--prompt", "-p", required=True, help="Description of the 3D model")
    parser.add_argument("--filename", "-f", help="Output filename (without extension)")
    parser.add_argument("--style", "-s", default="low-poly",
                       choices=["realistic", "cartoon", "low-poly", "sculpture"],
                       help="Art style for the model")
    parser.add_argument("--format", default="glb",
                       choices=["glb", "fbx", "obj"],
                       help="Output format")

    args = parser.parse_args()

    # Generate filename if not provided
    if not args.filename:
        timestamp = int(time.time())
        safe_name = args.prompt[:30].replace(" ", "-").replace("/", "-")
        args.filename = f"model-{safe_name}-{timestamp}"

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating 3D model: {args.prompt}")
    print(f"Style: {args.style}")

    # Step 1: Start generation
    print("  [1/3] Starting Meshy generation...")
    task_id = create_text_to_3d(args.prompt, args.style)
    print(f"  -> Task ID: {task_id}")

    # Step 2: Wait for completion
    print("  [2/3] Waiting for generation (this may take 1-3 minutes)...")
    result = wait_for_completion(task_id)

    # Step 3: Download model
    print("  [3/3] Downloading model...")
    model_urls = result.get("model_urls", {})

    # Get the requested format URL
    model_url = model_urls.get(args.format) or model_urls.get("glb")
    if not model_url:
        raise Exception(f"No download URL for format: {args.format}")

    output_path = ASSETS_DIR / f"{args.filename}.{args.format}"
    download_model(model_url, str(output_path))

    # Also download thumbnail if available
    thumbnail_url = result.get("thumbnail_url")
    if thumbnail_url:
        thumb_path = ASSETS_DIR / f"{args.filename}_thumb.png"
        download_model(thumbnail_url, str(thumb_path))
        print(f"  -> Thumbnail: {thumb_path}")

    print(f"\n3D Model saved to: {output_path}")
    print(f"MEDIA: {output_path}")

    # Print model info
    print(f"\nModel Info:")
    print(f"  Task ID: {task_id}")
    print(f"  Style: {args.style}")
    print(f"  Format: {args.format}")

if __name__ == "__main__":
    main()
