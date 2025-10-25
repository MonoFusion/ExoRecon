#!/bin/bash

# Variables to set
LOCAL_BASE_PATH="./raw_data"

# Fix locale warnings (optional)
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Get the local folder
LOCAL_FOLDER="${LOCAL_BASE_PATH}/indiana_music_14_3"

# Check if folder exists
if [ ! -d "${LOCAL_FOLDER}" ]; then
    echo "Error: Folder ${LOCAL_FOLDER} does not exist"
    exit 1
fi

echo "Processing: ${LOCAL_FOLDER}"

# Navigate to the local folder and run processing
echo "Running the specified command..."
cd "${LOCAL_FOLDER}"
# viewer_mps --vrs aria01.vrs --task vis --resize 512 --cam cam01
viewer_mps --vrs aria01.vrs --task mate --resize 512 --cam cam01


# Extract folder name for output
FOLDER_NAME=$(basename "${LOCAL_FOLDER}")

echo "Completed processing for folder: ${FOLDER_NAME}"