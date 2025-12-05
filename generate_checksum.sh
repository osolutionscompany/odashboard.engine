#!/bin/bash
# Generate SHA256 checksum for engine version
# Usage: ./generate_checksum.sh <version>
# Example: ./generate_checksum.sh 1.0.1

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./generate_checksum.sh <version>"
    echo "Example: ./generate_checksum.sh 1.0.1"
    exit 1
fi

ENGINE_FILE="versions/$VERSION/engine.py"

if [ ! -f "$ENGINE_FILE" ]; then
    echo "Error: File $ENGINE_FILE not found"
    exit 1
fi

echo "Generating SHA256 checksum for version $VERSION..."
CHECKSUM=$(shasum -a 256 "$ENGINE_FILE" | awk '{print $1}')

echo ""
echo "Checksum generated successfully!"
echo ""
echo "Add this to versions.json:"
echo ""
echo "  \"$VERSION\": {"
echo "    \"date\": \"$(date +%Y-%m-%d)\","
echo "    \"description\": \"Your description here\","
echo "    \"path\": \"$ENGINE_FILE\","
echo "    \"sha256\": \"$CHECKSUM\""
echo "  }"
echo ""
echo "SHA256: $CHECKSUM"
