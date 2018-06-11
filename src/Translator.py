import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Vocabulary import Vocabulary
from Models import NMT_RNNG
import re
import random
import math
import utils
import pickle
import os
import errno
import bleu


class Data(object):
    def __init__(self):
        self.src = []
        self.tgt = []
        self.action = []
        self.deprel = []
        self.trans = []     # output of decoder


class Translator(object):
    def __init__(self,
                 mode,
                 prepocessed,
                 srcVocaThreshold,
                 tgtVocaThreshold,
                 deprelLabelThreshold,
                 printEvery,
                 trainSize,
                 testSize,
                 devSize):
        if prepocessed:
            tgtTrain = './data/processed/train.en'
            actTrain = './data/processed/train.oracle.en'
            tgtDev = './data/processed/dev.en'
            actDev = './data/processed/dev.oracle.en'
            tgtTest = './data/processed/test.en'
            actTest = './data/processed/test.oracle.en'
            srcTrain = './data/processed/train.kr'
            deprelTrain = './data/processed/train.deprel.kr'
            srcDev = './data/processed/dev.kr'
            deprelDev = './data/processed/dev.deprel.kr'
            srcTest = './data/processed/test.kr'
            deprelTest = './data/processed/test.deprel.kr'
        else:
            train_permutation = list(range(0, 99999))
            random.shuffle(train_permutation)
            dev_permutation = list(range(0, 10000))
            random.shuffle(dev_permutation)
            print('Parsing target file into plain sentences & actions...')
            tgtTrain, actTrain = self.conll_to_action('./data/tagged_train.en', trainSize, train_permutation)
            tgtDev, actDev = self.conll_to_action('./data/tagged_dev.en', devSize, dev_permutation)
            print('Parsing source file into plain sentences & dependency relations...')
            srcTrain, deprelTrain = self.conll_to_deprels('./data/tagged_train.kr', trainSize, train_permutation)
            srcDev, deprelDev = self.conll_to_deprels('./data/tagged_dev.kr', devSize, dev_permutation)

        print('Loading processed data...')
        self.sourceVoc = Vocabulary(srcTrain, srcVocaThreshold, 'lang')
        self.targetVoc = Vocabulary(tgtTrain, tgtVocaThreshold, 'lang')
        self.actionVoc = Vocabulary(actTrain, None, 'action')
        self.deprelVoc = Vocabulary(deprelTrain, deprelLabelThreshold, 'deprel')
        self.trainData = []
        self.devData = []
        self.trainData = self.loadCorpus(srcTrain, tgtTrain, actTrain, deprelTrain,  self.trainData)
        self.devData = self.loadCorpus(srcDev, tgtDev, actDev, deprelDev, self.devData)
        self.printEvery = printEvery
        print('Loaded...')

    def train(self, criterion, NLL, optimizer, train=True):
        permutation = list(range(0, len(self.trainData)))
        random.shuffle(permutation)
        batchNumber = int(math.ceil(len(self.trainData) / self.miniBatchSize))
        for batch_i in range(1, batchNumber + 1):
            print('Progress: ' + str(batch_i) + '/' + str(batchNumber) + ' mini batches')
            startIdx = (batch_i - 1) * self.miniBatchSize
            endIdx = startIdx + self.miniBatchSize
            if endIdx > len(self.trainData):
                endIdx = len(self.trainData)
            indices = permutation[startIdx:endIdx]
            batch_trainData = [self.trainData[i] for i in indices]

            loss = 0
            optimizer.zero_grad()
            self.model.zero_grad()
            index = 0
            for data_in_batch in batch_trainData:
                index += 1

                train_src = torch.LongTensor(data_in_batch.src)
                train_tgt = torch.LongTensor(data_in_batch.tgt)
                train_action = torch.LongTensor(data_in_batch.action)
                train_deprel = data_in_batch.deprel

                src_length = len(data_in_batch.src)
                enc_hidden = self.model.enc_init_hidden()
                uts, s_tildes = self.model(train_src, train_tgt, train_action, train_deprel, src_length, enc_hidden)

                predicted_words = F.log_softmax(s_tildes.view(-1, len(self.targetVoc.tokenList)), dim=1)
                torch.set_printoptions(threshold=10000)
                # print(predicted_words[0])
                if index % self.printEvery == 0:
                    print("in batch "+str(batch_i)+", "+str(index)+"th data")
                    predictedWords = []
                    targetWords = []
                    print("source: ", end="")
                    for i in data_in_batch.src:
                        print(self.sourceVoc.tokenList[i][0], end=" ")
                    print("\ngold: ", end="")
                    for i in data_in_batch.tgt:
                        w = self.targetVoc.tokenList[i][0]
                        targetWords.append(w)
                        print(w, end=" ")
                    print("\ntarget: ", end="")
                    for i in range(list(predicted_words.shape)[0]):
                        topv, topi = predicted_words[i].topk(2)
                        if self.targetVoc.tokenList[topi[0]][0] == '*UNK*':
                            w = self.targetVoc.tokenList[topi[1]][0]
                        else:
                            w = self.targetVoc.tokenList[topi[0]][0]
                        predictedWords.append(w)
                        print(w, end=" ")
                    print("\n")
                loss_t = 0
                word_cnt = 0
                for i in range(min(list(predicted_words.shape)[0], len(train_tgt))):
                    loss_t += NLL(predicted_words[i].view(1, -1), torch.LongTensor([train_tgt[i]]))
                    word_cnt += 1
                if word_cnt != 0:
                    loss += loss_t / word_cnt

                # Backward(Action)
                loss += criterion(uts.view(-1, len(self.actionVoc.tokenList)), train_action)
            # dot = make_dot(uts, params=dict(self.model.named_parameters()))
            # with open("uts.dot", "w") as f:
            #     f.write(str(dot))
            # dot = make_dot(s_tildes)
            # with open("stildes.dot", "w") as f:
            #     f.write(str(dot))
            loss_val = round(loss.item(), 2) / len(batch_trainData)
            print("loss: ", loss_val)
            print("\nProcessing backward() and optimizer.step()...")
            loss.backward()
            optimizer.step()
            print("Done...")
        print("\nCompute BLEU score\n")
        self.computeBLEU()


    def computeBLEU(self):
        predictedSentences = []
        goldSentences = []
        for data in self.devData:
            dev_src = torch.LongTensor(data.src)
            dev_tgt = torch.LongTensor(data.tgt)
            dev_action = torch.LongTensor(data.action)
            dev_deprel = data.deprel
            src_length = len(data.src)
            enc_hidden = self.model.enc_init_hidden()
            uts, s_tildes = self.model(dev_src, dev_tgt, dev_action, dev_deprel, src_length, enc_hidden)
            predicted_words = F.log_softmax(s_tildes.view(-1, len(self.targetVoc.tokenList)), dim=1)
            predictedWords = []
            targetWords = []
            for i in data.tgt:
                w = self.targetVoc.tokenList[i][0]
                targetWords.append(w)
            for i in range(list(predicted_words.shape)[0]):
                topv, topi = predicted_words[i].topk(2)
                if self.targetVoc.tokenList[topi[0]][0] == '*UNK*':
                    w = self.targetVoc.tokenList[topi[1]][0]
                else:
                    w = self.targetVoc.tokenList[topi[0]][0]
                predictedWords.append(w)
            predictedSentences.append(predictedWords)
            goldSentences.append(targetWords)
            # torch.set_printoptions(threshold=10000)
        bleu_val = bleu.BLEU1(predictedSentences, goldSentences)
        print('BLEU1 score: ' + str(bleu_val))
        if bleu_val > self.model.prevPerp:
            self.model.prevPerp = bleu_val
            print('New high score, saving model to ' + self.modelPath)
            torch.save(self.model.state_dict(), self.modelPath)


    def loadCorpus(self, src, tgt, act, deprel, data):
        with open(src, encoding="utf-8") as f:
            for line in f:
                data.append(Data())
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '' and x != '?' and x != '.']
                for token in tokens:
                    if token in self.sourceVoc.tokenIndex:
                        data[-1].src.append(self.sourceVoc.tokenIndex[token])
                    else:
                        data[-1].src.append(self.sourceVoc.unkIndex)
                data[-1].src.append(self.sourceVoc.eosIndex)
        idx = 0
        with open(tgt, encoding="utf-8") as f:
            for line in f:
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '' and x != '?' and x != '.']
                for token in tokens:
                    if token in self.targetVoc.tokenIndex:
                        data[idx].tgt.append(self.targetVoc.tokenIndex[token])
                    else:
                        data[idx].tgt.append(self.targetVoc.unkIndex)
                data[idx].tgt.append(self.targetVoc.eosIndex)
                idx += 1
        idx = 0
        with open(act) as f:
            for line in f:
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '']
                if len(tokens) > 0:
                    if tokens[0] in self.actionVoc.tokenIndex:
                        data[idx].action.append(self.actionVoc.tokenIndex[tokens[0]])
                    else:
                        if "LEFT" in tokens[0]:
                            data[idx].action.append(self.actionVoc.tokenIndex['REDUCE-LEFT-ARC(unk)'])
                        elif "RIGHT" in tokens[0]:
                            data[idx].action.append(self.actionVoc.tokenIndex['REDUCE-RIGHT-ARC(unk)'])
                        else:
                            print("Error: Unknown word except shift/reduce.")
                else:
                    idx += 1
        idx = 0
        deprel_list = pickle.load(open(deprel, 'rb'))
        for dep_sen in deprel_list:
            for dep_word in dep_sen:
                label = dep_word[0]
                if label in self.deprelVoc.tokenIndex:
                    data[idx].deprel.append((self.deprelVoc.tokenIndex[label], dep_word[1], dep_word[2]))
                else:
                    data[idx].deprel.append((self.deprelVoc.unkIndex, dep_word[1], dep_word[2]))
            idx += 1
        return data

    def conll_to_action(self, tgt, num_, permutation):
        cnt = 0
        if 'dev' in tgt:
            oracle_fname = './data/processed/dev.oracle.en'
            txt_fname = './data/processed/dev.en'
        elif 'test' in tgt:
            oracle_fname = './data/processed/test.oracle.en'
            txt_fname = './data/processed/test.en'
        elif 'train' in tgt:
            oracle_fname = './data/processed/train.oracle.en'
            txt_fname = './data/processed/train.en'
        else:
            print('Error: invalid file name of ' + tgt)
            exit(1)
        oracle_f = open(oracle_fname, 'w', encoding='utf-8')
        plain_f = open(txt_fname, 'w', encoding='utf-8')
        tagged_file = open(tgt, 'r', encoding='utf-8')
        bulk = tagged_file.read()
        blocks = re.compile(r"\n{2,}").split(bulk)
        blocks = list(filter(None, blocks))
        for i in permutation:
            block = blocks[i]
            tokens = []
            buffer = []
            child_to_head_dict = {}
            for line in block.splitlines():
                attr_list = line.split('\t')
                if attr_list[1] == '.' or attr_list[1] == '?':
                    continue
                tokens.append(attr_list[1])
                num = int(attr_list[0])
                head = int(attr_list[6])
                label = attr_list[7]
                node = utils.Node(num, head, label)
                child_to_head_dict[num] = head
                buffer.append(node)
            arcs = utils.write_oracle(buffer, child_to_head_dict)
            for i, token in enumerate(tokens):
                token_lowered = token.lower()
                if i == 0:
                    plain_f.write(token_lowered)
                else:
                    plain_f.write(' ')
                    plain_f.write(token_lowered)
            plain_f.write('\n')
            for arc in arcs:
                oracle_f.write(arc + '\n')
            oracle_f.write('\n')
            cnt += 1
            if cnt == num_:
                break
        tagged_file.close()
        oracle_f.close()
        plain_f.close()
        return txt_fname, oracle_fname

    def conll_to_deprels(self, src, num_, permutation):
        cnt = 0
        if 'dev' in src:
            deprels_fname = './data/processed/dev.deprel.kr'
            txt_fname = './data/processed/dev.kr'
        elif 'test' in src:
            deprels_fname = './data/processed/test.deprel.kr'
            txt_fname = './data/processed/test.kr'
        elif 'train' in src:
            deprels_fname = './data/processed/train.deprel.kr'
            txt_fname = './data/processed/train.kr'
        else:
            print('Error: invalid file name of ' + src)
            exit(1)
        deprels_f = open(deprels_fname, 'wb')
        plain_f = open(txt_fname, 'w', encoding='utf-8')
        tagged_file = open(src, 'r', encoding='utf-8')
        bulk = tagged_file.read()
        blocks = re.compile(r"\n{2,}").split(bulk)
        blocks = list(filter(None, blocks))
        deprels = []
        for i in permutation:
            block = blocks[i]
            deprel = []
            tokens = []
            for line in block.splitlines():
                attr_list = line.split('\t')
                tokens.append(attr_list[1])
                num = int(attr_list[0])
                head = int(attr_list[6])
                label = attr_list[7]
                deprel.append((label, head, num))
            deprels.append(deprel)
            for i, token in enumerate(tokens):
                if i == 0:
                    plain_f.write(token)
                else:
                    plain_f.write(' ')
                    plain_f.write(token)
            plain_f.write('\n')
            cnt += 1
            if cnt == num_:
                break
        pickle.dump(deprels, deprels_f)
        tagged_file.close()
        deprels_f.close()
        plain_f.close()
        return txt_fname, deprels_fname

    def demo(self,
             inputDim,
             inputActDim,
             hiddenDim,
             hiddenEncDim,
             hiddenActDim,
             scale,
             miniBatchSize,
             learningRate,
             loadModel,
             modelDir,
             modelName,
             startIter,
             epochs,
             useGCN,
             gcnDim):
        try:
            os.makedirs(modelDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.modelPath = modelDir + modelName
        self.miniBatchSize = miniBatchSize
        self.model = NMT_RNNG(self.sourceVoc,
                              self.targetVoc,
                              self.actionVoc,
                              self.deprelVoc,
                              self.trainData,
                              self.devData,
                              inputDim,
                              inputActDim,
                              hiddenEncDim,
                              hiddenDim,
                              hiddenActDim,
                              scale,
                              self.miniBatchSize,
                              learningRate,
                              False,
                              useGCN,
                              gcnDim)
        if loadModel:
            self.model.load_state_dict(torch.load(self.modelPath))
        optimizer = optim.RMSprop(self.model.parameters(), lr=learningRate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.3, centered=True)
        criterion = nn.CrossEntropyLoss()
        NLL = nn.NLLLoss()
        print("# of Training Data:\t" + str(len(self.trainData)))
        print("# of Development Data:\t" + str(len(self.devData)))
        print("Source voc size: " + str(len(self.sourceVoc.tokenList)))
        print("Target voc size: " + str(len(self.targetVoc.tokenList)))
        print("Action voc size: " + str(len(self.actionVoc.tokenList)))
        print("Dependency Label voc size: " + str(len(self.deprelVoc.tokenList)))

        for i in range(epochs):
            print("Epoch " + str(i+startIter) + ' (lr = ' + str(self.model.learningRate) + ')')
            self.train(criterion, NLL, optimizer)






























