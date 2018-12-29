#!/bin/bash

cd /leap_lstm/ && chmod a+x start.sh && ./start.sh
#python leap_class_train.py


###root@zkl-Z97X:/home/zkl/zklcode/code# nvidia-docker run -it --privileged=true -v /home/zkl/zklcode/code/leap_lstm/:/leap_lstm tf-zkl:v2 /bin/bash -c "/leap_lstm/start.sh"
#/bin/bash: /leap_lstm/start.sh: Permission denied
