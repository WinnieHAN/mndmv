import subprocess
from multiprocessing import Pool
import os
import time
import numpy as np
import sys

langsPathes = np.array(['UD_Ancient_Greek', 'UD_Arabic', 'UD_Basque', 'UD_Bulgarian', 'UD_Croatian', 'UD_Czech',
                        'UD_Danish', 'UD_Dutch', 'UD_English', 'UD_Estonian', 'UD_Finnish', 'UD_French', 'UD_German',
                        'UD_Greek', 'UD_Hebrew', 'UD_Hindi', 'UD_Indonesian', 'UD_Italian', 'UD_Japanese',
                        'UD_Latin-ITTB', 'UD_Norwegian', 'UD_Persian', 'UD_Portuguese', 'UD_Slovenian', 'UD_Spanish',
                        'UD_Swedish'])
langs = np.array(
    ['grc', 'ar', 'eu', 'bg', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'de', 'el', 'he', 'hi', 'id', 'it', 'ja',
     'la_ittb', 'no', 'fa', 'pt', 'sl', 'es', 'sv'])


lang2i = {i: j for j, i in enumerate(langs)}

currPath = 'data/ud-treebanks-v1.4/'

idx = np.array([x for x in range(144)])


def strfind(a, b):
    return a[a.index(b):].split(' ')[1]


def Thread(arg):
    i = int(strfind(arg, '-idx'))
    print('idx:  ' + str(idx))
    cmd = arg[0:arg.index('-idx')]
    print('cmd:  ' + cmd)
    fname = "tuning7/dmv" + \
            "train-" + str(langs[i]) + \
            "-dev-" + str(langs[i]) + \
            "-id-" + str(i) + ".log"
    file = open(fname, 'w')
    subprocess.call(cmd, shell=True, stdout=file)


def main():
    arglist = []
    st = int(sys.argv[1])
    print(st)
    end = int(sys.argv[2])
    print(end)
    for i in range(st, end):
        trains = langs[i]
        dev = langs[i]
        pcmd = "python src/dmv_parser.py " + '--child_neural --em_type em --cvalency 2 --do_eval --ml_comb_type 1 --bidirectional --em_iter 1 --function_mask --non_neural_iter 60 --train ' + trains + ' --dev ' + dev + " -idx " + str(
            idx[i])
        # f_stc_train + " --dev " + f_stc_test + " --do_eval --split_epoch 2 --param_smoothing "+str(para_smooth[i])+" --split_factor 2 --cvalency 2 --em_type em --sample_batch 10000 --batch "+str(batch[i])+" --neural_epoch "+str(neural_epoch[i])+" --optim adam --em_iter 1 --epoch 50 --use_neural --lr "+str(lr[i])+" --function_mask --child_only " + "-idx " + str(idx[i])
        print(pcmd)
        arglist.append(pcmd)

    p = Pool(4)  # 20
    p.map(Thread, arglist, chunksize=1)
    p.close()
    p.join()


if __name__ == '__main__':
    main()
