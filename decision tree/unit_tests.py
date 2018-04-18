import ID3, parse, random
import matplotlib.pyplot as plt
import numpy as np



def testID3AndEvaluate():
  data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0))
    if ans != 1:
      print "ID3 test failed."
    else:
      print "ID3 test succeeded."
  else:
    print "ID3 test failed -- no tree returned"

def testPruning():
  data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  validationData = [dict(a=0, b=0, c=1, Class=1)]
  tree = ID3.ID3(data, 0)
  print "treesize is", ID3.treeSize(tree)
  ID3.prune(tree, validationData)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=1, c=1))
    if ans != 1:
      print "pruning test failed."
    else:
      print "pruning test succeeded."
  else:
    print "pruning test failed -- no tree returned."


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  fails = 0
  if tree != None:
    acc = ID3.test(tree, trainData)
    if acc == 1.0:
      print "testing on train data succeeded."
    else:
      print "testing on train data failed."
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print "testing on test data succeeded."
    else:
      print "testing on test data failed."
      fails = fails + 1
    if fails > 0:
      print "Failures: ", fails
    else:
      print "testID3AndTest succeeded."
  else:
    print "testID3andTest failed -- no tree returned."

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  withPruning = []
  withoutPruning = []
  data = parse.parse(inFile)
  for i in range(100):
    random.shuffle(data)
    train = data[:len(data)/2]
    valid = data[len(data)/2:3*len(data)/4]
    test = data[3*len(data)/4:]
  
    tree = ID3.ID3(train, 'democrat')
    acc = ID3.test(tree, train)
    print "training accuracy: ",acc
    acc = ID3.test(tree, valid)
    print "validation accuracy: ",acc
    acc = ID3.test(tree, test)
    print "test accuracy: ",acc
  
    ID3.prune(tree, valid)
    acc = ID3.test(tree, train)
    print "pruned tree train accuracy: ",acc
    acc = ID3.test(tree, valid)
    print "pruned tree validation accuracy: ",acc
    acc = ID3.test(tree, test)
    print "pruned tree test accuracy: ",acc
    withPruning.append(acc)
    tree = ID3.ID3(train + valid, 'democrat')
    acc = ID3.test(tree, test)
    print "no pruning test accuracy: ",acc
    withoutPruning.append(acc)
  print withPruning
  print withoutPruning
  print "average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning)

# add for test
def testWithoutPruningOnHouseData(inFile):
    withoutPruning = []
    data = parse.parse(inFile)
    for i in range(100):
        random.shuffle(data)
        train = data[:len(data) / 2]
        valid = data[len(data) / 2:3 * len(data) / 4]
        test = data[3 * len(data) / 4:]

        tree = ID3.ID3(train, 'democrat')
        acc = ID3.test(tree, train)
        print "training accuracy: ", acc
        acc = ID3.test(tree, valid)
        print "validation accuracy: ", acc
        acc = ID3.test(tree, test)
        print "test accuracy: ", acc

        tree = ID3.ID3(train + valid, 'democrat')
        acc = ID3.test(tree, test)
        print "no pruning test accuracy: ", acc
        withoutPruning.append(acc)
    print withoutPruning


# inFile - string location of the house data file
def producePlotDataTest(data, training_size):
    withPruning = []
    withoutPruning = []
    train_len = 7*(training_size/10);
    valid_len = 3*(training_size/10);
    for i in range(100):
        random.shuffle(data)
        train = data[:train_len]
        valid = data[train_len:train_len + valid_len]
        test = data[train_len + valid_len:]

        tree = ID3.ID3(train, 'democrat')
        ID3.prune(tree, valid)
        prune_acc = ID3.test(tree, test)
        print "pruned tree test accuracy: ", prune_acc
        withPruning.append(prune_acc)

        tree = ID3.ID3(train + valid, 'democrat')
        acc = ID3.test(tree, test)
        print "no pruning test accuracy: ", acc
        withoutPruning.append(acc)
    print withPruning
    print withoutPruning
    print "average with pruning", sum(withPruning) / len(withPruning), " without: ", sum(withoutPruning) / len(
        withoutPruning)
    return [sum(withPruning) / len(withPruning), sum(withoutPruning) / len(withoutPruning)]

def producePlotDataSet(inFile):

    data = parse.parse(inFile)
    prune = []
    nonprune = []

    for i in xrange(10, 301, 10):
        res = producePlotDataTest(data, i)
        prune.append(res[0])
        nonprune.append(res[1])

    x_step = np.arange(10, 301, 10)

    axes = plt.gca()
    axes.set_ylim([0.6, 1])

    plt.plot(x_step, prune, 'r--', x_step, nonprune, 'b--')
    plt.gca().legend(('average with pruning', 'average without pruning'), loc=4)
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy on test data')
    plt.title('Average accuracy of different training sizes')

    plt.show()




# testPruningOnHouseData('./house_votes_84.data')
#testID3AndEvaluate()
#testID3AndTest()
#testWithoutPruningOnHouseData('./house_votes_84.data')
# testPruningOnHouseData('./house_votes_84.data')
producePlotDataSet('./house_votes_84.data')