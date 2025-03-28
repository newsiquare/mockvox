#!/bin/bash
export REDIS_PASSWORD=$(openssl rand -base64 32)
sed -i.bak "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$REDIS_PASSWORD/" .env