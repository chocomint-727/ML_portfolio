#!/bin/bash

# Exit on error
set -e

# Check if argument provided
if [ -z "$1" ]; then
  echo "Usage: $0 <chunks_folder> [output_file]"
  echo "Example: $0 model.pt_chunks model.pt"
  exit 1
fi

FOLDER="$1"
OUTPUT="${2:-reconstructed_model.pt}"

# Check if folder exists
if [ ! -d "$FOLDER" ]; then
  echo "Error: Folder '$FOLDER' not found."
  exit 1
fi

# Find chunk prefix automatically
FIRST_FILE=$(ls "$FOLDER" | head -n 1)

if [ -z "$FIRST_FILE" ]; then
  echo "Error: No files found in folder."
  exit 1
fi

PREFIX=$(echo "$FIRST_FILE" | sed 's/[a-z][a-z]$//')

echo "Reconstructing file from chunks in '$FOLDER'..."
echo "Output file: $OUTPUT"

# Concatenate in correct order
cat "$FOLDER"/"$PREFIX"* > "$OUTPUT"

echo "Done."