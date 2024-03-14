docker build -t reid_train -f ./docker/Dockerfile-11.3 .
docker run -it --rm --gpus all --shm-size=80G -v $(pwd):/home/biplav/reid_train/ -v /home/biplav/qnap/VERI-1:/home/biplav/qnap/VERI-1 reid_train /bin/bash