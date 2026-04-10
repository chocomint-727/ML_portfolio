#!/bin/bash

# Exit on error
set -e

# Check if argument provided
if [ -z "$1" ]; then
  echo "Usage: $0 <file_to_split> [chunk_size]"
  echo "Example: $0 model.pt 50M"
  exit 1
fi

FILE="$1"
CHUNK_SIZE="${2:-50M}"  # Default to 50MB

# Check if file exists
if [ ! -f "$FILE" ]; then
  echo "Error: File '$FILE' not found."
  exit 1
fi

BASENAME=$(basename "$FILE")
FOLDER="${BASENAME}_chunks"

# Create output folder
mkdir -p "$FOLDER"

echo "Splitting '$FILE' into chunks of size $CHUNK_SIZE..."
echo "Output folder: $FOLDER"

# Perform split into the folder
split -b "$CHUNK_SIZE" "$FILE" "$FOLDER/${BASENAME}_part_"

echo "Done."
echo "Chunks created in $FOLDER:"
ls "$FOLDER"