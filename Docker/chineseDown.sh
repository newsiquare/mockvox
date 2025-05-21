#!/usr/bin/env bash

echo "Downloading models..."
aria2c --disable-ipv6 --input-file /mockvox/Docker/links.txt --dir /mockvox --continue