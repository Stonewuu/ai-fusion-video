#!/bin/sh
set -eu

node /app/docker/write-runtime-config.mjs
exec node /app/server.js
