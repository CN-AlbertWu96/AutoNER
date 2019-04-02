from tqdm import tqdm
import re
from collections import defaultdict
import json

def addNewLine(match):
    string = match.group()
    return ' ' + string + " "

def getRawText(file_path, out_file_path):
    with open(file_path, 'r') as fin:
        lines = fin.readlines()

    with open(out_file_path, 'w') as fout:
        for i in tqdm(range(20000)):
            line = lines[i].strip()
            if line == "-doc-start-":
                continue
            line = re.sub(r"[!\"#$%&'()*+,./:;<=>?@\[\\\]^`{\|}]", addNewLine, line)
            words = line.split()
            for i in range(len(words)):
                if words[i][0].isalpha():
                    words[i] = words[i][0].upper() + words[i][1:]
                    break
            fout.write("\n".join(words) + "\n\n")

def writeChunkerwithSafety(file, lines):
    with open(file, 'w') as fout:
        sentence = ["<s> O None S\n"]
        for i in tqdm(range(len(lines))):
            line = lines[i].strip()
            # print('[%s]'%i, bool(line), line)
            if "DOCSTART" in line:
                continue
            if not line:
                if len(sentence) > 1:
                    sentence.append("<eof> I None S\n\n\n")
                    fout.write(''.join(sentence))
                sentence = ["<s> O None S\n"]
            else:
                data = line.strip().split()
                word = data[0]
                try:
                    sep, entity = data[3].split('-')
                    if entity == "MISC":
                        entity = "None"
                except:
                    # print(line)
                    sep, entity = data[3], "None"
                # IOBES
                if sep in ["O", "B", "S"]:
                    sep = "I"
                else:
                    sep = "O"
                sentence.append("%s %s %s S\n"%(word, sep, entity))

def writeChunker(file, lines):
    with open(file, 'w') as fout:
        sentence = ["<s> O None\n"]
        for i in tqdm(range(len(lines))):
            line = lines[i].strip()
            # print('[%s]'%i, bool(line), line)
            if "DOCSTART" in line:
                continue
            if not line:
                if len(sentence) > 1:
                    sentence.append("<eof> I None\n\n\n")
                    fout.write(''.join(sentence))
                sentence = ["<s> O None\n"]
            else:
                data = line.strip().split()
                word = data[0]
                try:
                    sep, entity = data[3].split('-')
                    if entity == "MISC":
                        entity = "None"
                except:
                    sep, entity = data[3], "None"
                # IOBES
                if sep in ["O", "B", "S"]:
                    sep = "I"
                else:
                    sep = "O"
                sentence.append("%s %s %s\n"%(word, sep, entity))

def getTestChunker(dev_path, test_path, dev_out_file_path, test_out_file_path):
    with open(dev_path, 'r') as fin:
        dev_lines = fin.readlines()

    with open(test_path, 'r') as fin:
        test_lines = fin.readlines()

    writeChunker(dev_out_file_path, dev_lines)
    writeChunker(test_out_file_path, test_lines)

def getDictCore(dic_path, dic_out_path):
    with open(dic_path, 'r') as fin:
        lines = fin.readlines()

    d = defaultdict(int)
    with open(dic_out_path, 'w') as fout:
        for i in tqdm(range(len(lines))):
            line = lines[i].strip()
            data = line.split('\t')
            canonical_name = data[0]
            alias_name = json.loads(data[1])
            entity_types = json.loads(data[2])
            x = 0
            if 'person' in entity_types:
                d['person'] += 1
                fout.write('%s\t%s\n'%('PER',canonical_name.replace('_',' ')))
                x += 1
            if 'location' in entity_types:
                d['location'] += 1
                entity_type = 'LOC'
                fout.write('%s\t%s\n'%('LOC',canonical_name.replace('_',' ')))
                x += 1
            if 'organization' in entity_types:
                d['organization'] += 1
                entity_type = 'ORG'
                fout.write('%s\t%s\n'%('ORG',canonical_name.replace('_',' ')))
                x += 1
            if x>=2:
                print(line)
    print(d)

def getDictFull(dic_path, dic_out_path):
    with open(dic_path, 'r') as fin:
        lines = fin.readlines()

    with open(dic_out_path, 'w') as fout:
        count = 0
        for i in tqdm(range(len(lines))):
            line = lines[i].strip()
            data = line.split('\t')
            canonical_name = data[0]
            alias_name = json.loads(data[1])
            alias_name.insert(0, canonical_name)
            for name in set(alias_name):
                count += 1
                fout.write(name.replace('_',' ') + '\n')
        print('Total number: %s'%count)

def getGoldenChunker(train_path, train_out_file_path):
    with open(train_path, 'r') as fin:
        train_lines = fin.readlines()
    writeChunkerwithSafety(train_out_file_path, train_lines)

def collectData(lines, set):
    queue = list()
    for i in tqdm(range(len(lines))):
        line = lines[i].strip()
        data = line.strip().split()
        if not (line.isspace() or len(data) < 4):
            type_dic = {'LOC':0, 'ORG':1, 'PER':2}
            token, sep, type = data[0], data[1], data[2]
            if token == '<s>':
                continue
            if sep == 'I' and queue:
                phrase, tps = '', ''
                while queue:
                    word, tps = queue.pop(0)
                    phrase += ' ' + word 
                phrase = phrase.lower().strip()
                if tps:
                    for tp in tps.split(','):
                        set[type_dic[tp]].add(phrase)
            if type in type_dic:
                queue.append((token, type))

def countOverlap(gold_path, dic_path):
    with open(gold_path, 'r') as fin:
        lines_gold = fin.readlines()
    with open(dic_path, 'r') as fin:
        lines_dis = fin.readlines()
    set_gold = [set(), set(), set()]
    set_dis = [set(), set(), set()]
    collectData(lines_gold, set_gold)
    collectData(lines_dis, set_dis)
    type_index = {0:'LOC', 1:'ORG', 2:'PER'}
    for i, (set1, set2) in enumerate(zip(set_gold, set_dis)):
        intersection = set1.intersection(set2)
        print('%s: %s-%s-%s'%(type_index[i], len(set1)-len(intersection), len(intersection), len(set2)-len(intersection)))


if __name__ == "__main__":
    dataset = "CoNLL03"
    # model = "CoNLL03_newdic"
    # gold_path = '../data/%s/golden.ck'%dataset
    # dic_path =  '../models/%s/annotations.ck'%(model)
    # countOverlap(gold_path, dic_path)
    # raw_path = '../../%s/CoNLL03_all_english_news.txt'%dataset
    # raw_out_file_path = '../data/%s/raw_text.txt'%dataset
    # getRawText(raw_path, raw_out_file_path)

    dev_path, test_path = '../../%s/testa.iobes'%dataset, '../../%s/testb.iobes'%dataset
    dev_out_file_path, test_out_file_path = '../data/%s/truth_dev.ck'%dataset, '../data/%s/truth_test.ck'%dataset
    getTestChunker(dev_path, test_path, dev_out_file_path, test_out_file_path)
    
    # dic_core_path, dic_core_out_file = '../../%s/YOGA_PER_LOC_ORG.tsv'%dataset, '../data/%s/dict_core.txt'%dataset
    # getDictCore(dic_core_path, dic_core_out_file)
    
    # dic_full_path, dic_full_out_path = '../../%s/YOGA_PER_LOC_ORG.tsv'%dataset, '../data/%s/dict_full.txt'%dataset
    # getDictFull(dic_full_path, dic_full_out_path)

    # train_path = '../../%s/train.iobes'%dataset
    # train_out_file_path = '../data/%s/golden.ck'%dataset
    # getGoldenChunker(train_path, train_out_file_path)
