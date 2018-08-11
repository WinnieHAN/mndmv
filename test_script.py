import os
import sys

num_iter = int(sys.argv[1])

for i in range(num_iter):
    command = "python test.py --idx " + str(
        i + 1) + " --outdir /Users/wangge/Documents/workspace/ml_dmv/output --test /Users/wangge/Documents/workspace/ml_dmv/data/wsj10_te --model /Users/wangge/Documents/workspace/ml_dmv/output/dmv.model --log /Users/wangge/Documents/workspace/ml_dmv/output/log"
    os.system(command)
