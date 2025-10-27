#!/bin/bash

echo "If the container is running but a browser tab is not opened, please navigate to http://localhost:8765."

docker run --rm --init --name pgsui \
    -p 8765:8765 \
    -v "$PWD":/work \
    -e TZ=Etc/UTC \
    pgsui-web:1.0
