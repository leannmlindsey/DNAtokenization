#!/bin/bash

# Save the starting directory
start_dir=$(pwd)

# Function to find max sequence length in train.csv
find_max_length() {
    if [ -f "train.csv" ]; then
        echo "Finding max sequence length in $(pwd)/train.csv"
        max_length=$(awk -F, 'NR>1 {print length($1)}' train.csv | sort -nr | head -1)
        echo "Max sequence length: $max_length"
    else
        echo "No train.csv file found in $(pwd)"
    fi
    echo "----------------------------------------"
}

# Check if directory argument was provided
if [ -n "$1" ]; then
    cd "$1" || { echo "Could not change to directory $1"; exit 1; }
fi

# Process the current directory first
echo "Current directory: $(pwd)"
find_max_length

# Process each subdirectory
for dir in */; do
    if [ -d "$dir" ]; then
        echo "Entering directory: $dir"
        cd "$dir" || continue
        echo "Current directory: $(pwd)"
        find_max_length
        cd "$start_dir" || exit
        if [ -n "$1" ]; then
            cd "$1" || exit
        fi
    fi
done

echo "Processing complete"
