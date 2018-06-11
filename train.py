import sys
sys.path.insert(0, './src')
import Translator

#   trainSize < 100000
#   testSize < 10001
#   devSize < 10001
t = Translator.Translator(mode='train',
                          prepocessed=False,
                          srcVocaThreshold=0,
                          tgtVocaThreshold=0,
                          deprelLabelThreshold=0,
                          printEvery=50,
                          trainSize=99999,
                          testSize=10000,
                          devSize=10000)

t.demo(inputDim=128,
       inputActDim=32,
       hiddenDim=128,
       hiddenEncDim=128,
       hiddenActDim=16,
       scale=0,
       miniBatchSize=256,
       learningRate=0.01,
       loadModel=False,
       modelDir='./data/saved_model/',
       modelName='30000-10000-10-10-5.pt',
       startIter=1,
       epochs=30,
       useGCN=True,
       gcnDim=128)































