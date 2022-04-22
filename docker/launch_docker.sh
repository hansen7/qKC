#!/bin/bash

docker run -it \
	--rm \
	--shm-size=1g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v "$(dirname $PWD):/workspace/qkc" \
	-v "/scratch/hw501/data_source/:/scratch/hw501/data_source/" \
	-v "/scratches/mario/hw501/data_source:/scratches/mario/hw501/data_source/" \
	qkc bash

# --runtime=nvidia \
# -v + any external directories if you are using them
