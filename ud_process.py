import sys
import utils
import os

dir_path = sys.argv[1]
target_path = sys.argv[2]
length = int(sys.argv[3])
for dir in os.listdir(dir_path):
    file_list = os.listdir(dir_path + '/' + dir)
    for file in file_list:
        leng = len(file)
        if leng < 18:
            continue
        key = file[leng - 16]
        last_key = file[leng - 17]
        last_last_key = file[leng - 18]
        train_key = file[leng - 8]
        if (key == 'v' and last_key == 'e') or (
                    key == 't' and last_key == 's' and last_last_key == 'e') or train_key == '0':
            _, _, sentences = utils.read_data(dir_path + '/' + dir + '/' + file, False)
            filtered_sentences = []
            for s in sentences:
                if s.size - 1 < length + 1:
                    filtered_sentences.append(s)
            fw = open(target_path + '/' + file, "w")
            print 'writing for ' + file
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
            print 'writing for ' + file + ' completed'
            fw.close()
print("writing completed")
