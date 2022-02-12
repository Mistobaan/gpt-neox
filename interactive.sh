#!/bin/sh

# nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0,1 --shm-size=1g --ulimit memlock=-1 \
#   --mount type=bind,src=$PWD,dst=/gpt-neox \
#   --volume /tmp/logs=/gpt-neox/logs \
#   --user mchorse --workdir /gpt-neox gpt-neox  \

python ./deepy.py \
  generate.py --conf_dir ./configs 20B-dual-gpu.yml 
# text_generation.yml

# export DEEPERSPEED=1
# python ./deepy.py \
#   generate.py --conf_dir ./configs 20B-dual-gpu.yml text_generation.yml

# deepspeed --num_gpus 2 --num_nodes 1 \
#   inference.py --conf_dir ./configs 20B-dual-gpu.yml text_generation.yml