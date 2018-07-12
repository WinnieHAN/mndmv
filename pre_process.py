import sys
import utils

file_path = sys.argv[1]
target_path = sys.argv[2]
length = int(sys.argv[3])
_,_,sentences = utils.read_data(file_path, False)
filtered_sentences = []
for s in sentences:
    if s.size-1<length+1:
        filtered_sentences.append(s)
fw = open(target_path,"w")
for s in filtered_sentences:
    for t in s.entries:
        if t.id == 0:
            continue
        fw.write(str(t.id))
        fw.write('\t')
        fw.write(t.form)
        fw.write('\t')
        fw.write(t.lemma)
        fw.write('\t')
        fw.write(t.pos)
        fw.write('\t')
        fw.write(t.cpos)
        fw.write('\t')
        fw.write(t.feats)
        fw.write('\t')
        fw.write(str(t.parent_id))
        fw.write('\t')
        fw.write(t.relation)
        fw.write('\t')
        fw.write(t.deps)
        fw.write('\t')
        fw.write(t.misc)
        fw.write('\n')


    fw.write('\n')
fw.close()
print("writing completed")


