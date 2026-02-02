# Game Forge

AI-powered game development system using Godot, with automated asset generation.

## Features

- **2D Sprite Generation** - Gemini Pro with Imagen 3 (Nano Banana)
- **3D Model Generation** - Meshy API
- **Code Generation** - GDScript for Godot 4.x
- **Orchestrated by Clawdbot** - Natural language game development

## Directory Structure

```
game-forge/
├── projects/          # Individual game projects (Godot)
├── assets/
│   ├── sprites/       # Generated 2D assets
│   └── models/        # Generated 3D models
├── templates/         # Project templates
├── tools/             # Asset generation scripts
└── scripts/           # Automation scripts
```

## APIs Used

- **Gemini Browser API** (localhost:8891) - 2D image generation
- **Meshy API** - 3D model generation

## Usage

Clawdbot can create games via natural language:
- "Create a 2D platformer with a robot character"
- "Generate a sprite for a health potion"
- "Add an enemy that shoots fireballs"

## Development

Games are developed in the sandbox VM (clawdbot-sandbox) with Godot installed.
