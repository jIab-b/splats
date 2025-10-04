#!/bin/bash

# Launch Godot Capture for Chinch GLTF - All-in-One Script
# Run from /home/beed/splats/ or anywhere; auto-cds to project root.

PROJECT_ROOT="/home/beed/splats"
GODOT_EXEC="./Godot4.x86_64"
MAIN_SCENE="res://godot_stuff/auto_capture.tscn"

# Change to project root
cd "$PROJECT_ROOT" || { echo "Failed to cd to $PROJECT_ROOT"; exit 1; }

# Check if Godot executable exists
if [ ! -f "$GODOT_EXEC" ]; then
    echo "Godot executable not found at $PROJECT_ROOT/$GODOT_EXEC"
    echo "Ensure Godot4.x86_64 is in the project root."
    exit 1
fi

# Check if main scene exists
if [ ! -f "godot_stuff/auto_capture.tscn" ]; then
    echo "Main scene not found at $PROJECT_ROOT/godot_stuff/auto_capture.tscn"
    echo "Run Godot editor first to generate it or check paths."
    exit 1
fi

# Check if GLTF exists
if [ ! -f "scenes/chinch/scene.gltf" ]; then
    echo "GLTF not found at $PROJECT_ROOT/scenes/chinch/scene.gltf"
    echo "Copy your chinch GLTF there first."
    exit 1
fi

# Launch Godot in runtime/play mode (no editor)
echo "Launching Godot capture session for chinch..."
echo "Navigate with WASD/mouse, spam 6 to capture, 7 to save transforms.json"
echo "Outputs to data/chinch/"

"$GODOT_EXEC" --path . --main-scene "$MAIN_SCENE" --no-editor

# Post-launch: Optional cleanup or message
echo "Session ended. Check data/chinch/ for captures."
