from operator import itemgetter
import re
import pickle

class Vocabulary(object):
    def __init__(self, trainFile, tokenFreqThreshold, Action_or_Language):
        if Action_or_Language == 'lang':
            self.initLang(trainFile, tokenFreqThreshold)
        elif Action_or_Language == 'action':
            self.initAction(trainFile)
        else:
            self.initDeprel(trainFile, tokenFreqThreshold)

    def initLang(self, trainFile, tokenFreqThreshold):
        self.tokenList = []     # list of (token, count)
        self.tokenIndex = {}    # token to index
        unkCount = 0
        eosCount = 0
        tokenCount = {}
        with open(trainFile, encoding="utf-8") as f:
            for line in f:
                eosCount += 1
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '' and x != '.' and x != '?']
                for token in tokens:
                    if token in tokenCount:
                        tokenCount[token] += 1
                    else:
                        tokenCount[token] = 1
        for token, count in tokenCount.items():
            if count >= tokenFreqThreshold:
                self.tokenList.append((token, count))
            else:
                unkCount += count
        self.tokenList.sort(key=itemgetter(1))
        self.tokenList.reverse()    # (token, count) tuple이 count가 큰 순서로 정렬
        tokenList_len = len(self.tokenList)
        for i in range(tokenList_len):
            self.tokenIndex[self.tokenList[i][0]] = i
        self.eosIndex = tokenList_len
        self.tokenList.append(("*EOS*", eosCount))
        self.unkIndex = tokenList_len+1
        self.tokenList.append(("*UNK*", unkCount))

    def initAction(self, trainFile):
        self.tokenList = []  # list of (token, count)
        self.tokenIndex = {}  # token to index
        tokenCount = {}
        with open(trainFile) as f:
            for line in f:
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '']
                for token in tokens:
                    if token in tokenCount:
                        tokenCount[token] += 1
                    else:
                        tokenCount[token] = 1
        for token, count in tokenCount.items():
            if "SHIFT" in token:
                self.tokenList.append((token, count, 0))
            elif "LEFT" in token:
                self.tokenList.append((token, count, 1))
            elif "RIGHT" in token:
                self.tokenList.append((token, count, 2))
            else:
                print("Error: Non shift/reduce word.")
        self.tokenList.append(('REDUCE-RIGHT-ARC(unk)', 0, 2))
        self.tokenList.append(('REDUCE-LEFT-ARC(unk)', 0, 1))
        self.tokenList.sort(key=itemgetter(1))
        self.tokenList.reverse()  # (token, count, actionIndex) tuple이 count가 큰 순서로 정렬
        tokenList_len = len(self.tokenList)
        for i in range(tokenList_len):
            self.tokenIndex[self.tokenList[i][0]] = i

    def initDeprel(self, deprel, deprelLabelThreshold):
        self.tokenList = []
        self.tokenIndex = {}
        self.dirList = []
        tokenCount = {}
        unkCount = 0
        deprel_list = pickle.load(open(deprel, 'rb'))
        for dep_in_sen in deprel_list:
            for dep_word in dep_in_sen:
                label = dep_word[0]
                if label in tokenCount:
                    tokenCount[label] += 1
                else:
                    tokenCount[label] = 1
        for token, count in tokenCount.items():
            if count >= deprelLabelThreshold:
                self.tokenList.append((token, count))
            else:
                unkCount += count
        self.tokenList.sort(key=itemgetter(1))
        self.tokenList.reverse()
        tokenList_len = len(self.tokenList)
        self.unkIndex = tokenList_len
        self.tokenList.append(("unk", unkCount))
        for i in range(tokenList_len):
            self.tokenIndex[self.tokenList[i][0]] = i
        deprel_list = pickle.load(open(deprel, 'rb'))
        for dep_in_sen in deprel_list:
            for dep_word in dep_in_sen:
                label = dep_word[0]
                if label in self.tokenIndex:
                    labelIdx = self.tokenIndex[label]
                else:
                    labelIdx = self.unkIndex
                item = (labelIdx, dep_word[1], dep_word[2])
                self.dirList.append(item)

























