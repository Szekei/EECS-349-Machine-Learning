class Node:
    def __init__(self):
        self.attribute = None
        self.children = {}
        self.childrenBuffer = {}
        self.label = None
        self.isLeaf = False
        self.trainingClassCounter = 0
        self.leafClass = {}

    # you may want to add additional fields here...

    def __init__(self, attribute, children, label, isLeaf=False, trainingCounter=0):
        self.attribute = attribute
        self.children = children
        self.label = label
        self.isLeaf = isLeaf
        self.trainingClassCounter = trainingCounter
        self.leafClass = {}

    def get_attribute(self):
        return self.attribute

    def get_child(self, value):
        if self.children.has_key(value):
            return self.children[value]
        return None

    def get_children(self):
        return self.children
    
    def get_training_count(self):
        return self.trainingCounter

    def get_label(self):
        return self.label

    def add_child(self, child_node):
        self.children[child_node.get_attribute()] = child_node
        
    def set_to_leaf(self, label):
        self.childrenBuffer = self.children
        self.children = {}
        self.isLeaf = True
        self.label = label
        
    def recover_to_node(self):
        if self.isLeaf is False:
            return
        self.children = self.childrenBuffer;
        self.childrenBuffer = {}
        self.isLeaf = False
        self.label = None
