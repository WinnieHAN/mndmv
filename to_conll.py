import random
import sys


def read_4_lines():
    global f_r
    one_sentence = []
    for i in range(0, 4):
        a = f_r.readline()
        one_sentence.append(a)
    return one_sentence

data_r = sys.argv[1]
data_w = sys.argv[2]
all_sentence_read = []
f_r = open(data_r, 'r')
one_sentence = read_4_lines()
f_r.readline()
all_sentence_read.append(one_sentence)
while (one_sentence[0] != ""):
    one_sentence = read_4_lines()
    if (one_sentence[0] != ""):
        all_sentence_read.append(one_sentence)
        f_r.readline()
f_r.close()
f_w = open(data_w, 'w')

for i in range(len(all_sentence_read)):
    s = all_sentence_read[i]
    toks = s[0].split()
    pos = s[1].split()
    types = s[3].split()
    deps = s[2].split()
    for l in range(len(toks)):
        f_w.write(str(l + 1))
        f_w.write("\t")
        f_w.write(toks[l])
        f_w.write("\t")
        f_w.write("-")
        f_w.write("\t")
        f_w.write(pos[l])
        f_w.write("\t")
        f_w.write("-")
        f_w.write("\t")
        f_w.write("-")
        f_w.write("\t")
        f_w.write(types[l])
        f_w.write("\t")
        f_w.write(deps[l])
        f_w.write("\t")
        f_w.write("-")
        f_w.write("\t")
        f_w.write("-")
        f_w.write("\t")
        f_w.write("\n")
    f_w.write("\n")
f_w.close()
