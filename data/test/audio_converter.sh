#!/bin/bash
set -eou pipefail

if [ -z $1 ]; then
    echo "Please supply audio directory with files to strip"
fi

cwd=$(pwd)
echo "Files will be looked up under" "${1}", "output will be in current directory!" "${cwd}" 

for file in "${1}"/*.mp3; do
    echo "Processing ${file}" "..."
    outname=$(basename "${file}")
    ffmpeg -i "${file}" -ss 0 -t 60 "${outname}"
done

echo "Processing complete!"
