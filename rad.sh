#!/bin/bash
if ! [ -d "/data/$1" ] || [ -z "$1" ] ; then
  echo "❌ Could not find data folder";
  echo "Usage: bash rad.sh data_folder";
  exit 1;
fi
echo "📈 Running analysis...";

python rad_main.py -p "/data/$1"