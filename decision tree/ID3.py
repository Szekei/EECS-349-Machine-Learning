from node import Node
import math


def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    processMissingData(examples)
    values = [record['Class'] for record in examples]
    if len(examples) == 0:
        return Node('Class', {}, default)

    elif values.count(values[0]) == len(values):
        return Node('Class', {}, values[0])

    # in case examples only contain the Class labels, return directly
    elif len(examples[0]) == 1:
        return Node('Class', {}, default)

    else:

        best = chooseBestAttribute(examples)

        root = Node(best, {}, None)

        for value in getValuesByKey(examples, best):
            newExamples = getNewExamples(examples, best, value)
            child = ID3(newExamples, mostFreqAttr(newExamples, 'Class'))
            root.children[value] = child

    return root


def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    processMissingData(examples)
    correct = 0.0
    for example in examples:
        if example['Class'] == evaluate(node, example):
            correct += 1.0
    return correct/len(examples)

def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    while len(node.get_children()) != 0:
        label = example[node.get_attribute()]
        if node.get_children().has_key(label):
            node = node.get_child(label)
        # in case the training data doesn't contain the label, return the example's Class label
        else:
            return example['Class']

    return node.get_label()

# choose best attribute with max infogain
def chooseBestAttribute(examples):
    best = examples[0].keys()[0]
    if best == 'Class' and len(examples[0].keys()) > 1:
        best = examples[0].keys()[1]

    max_infogain = 0.0

    for key in examples[0].keys():
        if key != 'Class':
            infogain = infoGain(examples, key)
            if infogain > max_infogain:
                max_infogain = infogain
                best = key

    return best


def infoGain(examples, currentAttr):
    valueMap = {}
    childEntropy = 0.0

    values = [record[currentAttr] for record in examples]

    for v in values:
        if valueMap.has_key(v):
            valueMap[v] += 1.0
        else:
            valueMap[v] = 1.0

    for value in valueMap.keys():
        prob = valueMap[value] / len(values)
        childData = [example for example in examples if example[currentAttr] == value]
        childEntropy += prob * entropy(childData)

    priorEntropy = entropy(examples)
    return priorEntropy - childEntropy


def entropy(data):
    values = [record['Class'] for record in data]
    valueMap = {}
    entropy = 0.0
    for v in values:
        if valueMap.has_key(v):
            valueMap[v] += 1.0
        else:
            valueMap[v] = 1.0
    for v in valueMap.values():
        entropy += -(v / len(values)) * math.log(v / len(values), 2)

    return entropy


def getNewExamples(examples, best, value):
    newExamples = []
    for entry in examples:
        if entry[best] == value:
            newEntry = {}
            for key, val in entry.iteritems():
                if key != best:
                    newEntry[key] = val
            newExamples.append(newEntry)
    return newExamples


def getValuesByKey(examples, key):
    values = []
    for entry in examples:
        if entry[key] not in values:
            values.append(entry[key])
    return values


def mostFreqAttr(examples, attr):

    valueMap = {}

    values = [record[attr] for record in examples]

    for v in values:
        if valueMap.has_key(v):
            valueMap[v] += 1.0
        else:
            valueMap[v] = 1.0

    max_attr = ''
    max_freq = 0
    for key in valueMap.keys():
        if valueMap[key] > max_freq:
            max_freq = valueMap[key]
            max_attr = key
    return max_attr

# replace the missing element with most frequent label
def processMissingData(examples):

    valueMap = {}

    for key in examples[0].keys():
        valueMap[key] = mostFreqAttr(examples, key)

    for example in examples:
        for key in example.keys():
            if example[key] == '?':
                example[key] = valueMap[key]
    return examples