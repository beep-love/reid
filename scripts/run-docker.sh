docker build -t reid_train -f ./docker/Dockerfile .
docker run -it --rm --gpus all -v $(pwd):/home/biplav/reid_train/ -v /home/biplav/qnap/VERI-1:/home/biplav/qnap/VERI-1 reid_train /bin/bash