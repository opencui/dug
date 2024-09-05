#!/bin/bash

# this shell can expand the tar ball, and copy the extracted file to right place for debug.
# Check if a filename is provided
if [ $# -eq 0 ]; then
    echo "Please provide a tar.gz filename as an argument."
    exit 1
fi

# Create a temporary directory
temp_dir=$(mktemp -d)

# Extract the tar.gz file to the temporary directory
tar -xzf "$1" -C "$temp_dir"

files=$(find "$temp_dir/en" -name "me_test*")

echo "Files found (simple method):"
echo "$files"

file_count=$(echo "$files" | wc -l)

# Test if there's exactly one match
if [ "$file_count" -ne 1 ]; then
    echo "Error: Number of files found is not 1."
    echo "Actual number of files found: $file_count"
    exit 1
fi

filename=$(basename "${files[0]}")
IFS='_' read -ra parts1 <<< "$filename"
last_element="${parts1[@]: -1}"
filename_without_ext="${last_element%.kt}"

path="me.test_${filename_without_ext}_en_746395988637257728"

mkdir -p "$path"

echo "Creating $path"

cp  "$temp_dir/en/dumeta"/* "$path"

url="127.0.0.1:3001/v1/index/${path}"
echo "curl $url"
curl "$url"

# Clean up: remove the temporary directory
rm -rf "$temp_dir"

echo "Processing complete"
