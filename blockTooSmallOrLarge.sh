#!/bin/bash

if [ $# -lt 2 ] ; then
   echo "Run with: $0 <minSize> <maxSize> [blocklistname]"
   exit 1
fi

minSize=$1
maxSize=$2
shift
shift

blocklistfile="blocklistTooSmallOrLarge.txt"
if [ $# -eq 1 ] ; then
    blocklistfile="$1"
fi

echo "Processing files ..." 1>&2
echo "# Blocking files too large or too small" >> "$blocklistfile"
while read -r line ; do
   # This is super slow, but much easier than attempting to write in Bash Regex match logic
   # Despite the slowness, it is plenty fast for its intended purpose
   size=$(echo "$line" | cut -d\} -f 4 | cut -d, -f3)
   if [ $size -lt $minSize ] || [ $size -gt $maxSize ] ; then
      echo "$line" >> "$blocklistfile"
   fi
done < <(find . -name "*tiff" | cut -c3- )
echo "Completed processing files." 1>&2
echo
