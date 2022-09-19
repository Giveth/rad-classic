#!/bin/bash  

if ! [ -d "$1" ] ; then
  echo "❌ Could not find data folder";
  echo "Usage: bash start.sh [absoute path to root data folder]";
  exit 1;
fi

IMAGE=$( docker images | grep rad )
if [ -z "$IMAGE" ]
then
  echo "🔃 Building image..."
  docker build --tag rad .
else
  echo "✅ Using built image."
fi

CONTAINER=$( docker ps | grep rad | cut -d' ' -f1 )
if [ -z "$CONTAINER" ]
then
  echo "🔃 Starting container..."
else
  echo "✅ Restarting container..."
  docker stop $CONTAINER
fi

docker run --rm -d -it -v $(pwd):/app -v "$1:/data" rad

echo "✅ Opening RAD shell…" 
echo "💡 Type 'bash rad.sh' to run analysis/distribution."
echo "💡 Type 'exit' to quit."
docker exec -ti $( docker ps | grep rad | cut -d' ' -f1 ) bash