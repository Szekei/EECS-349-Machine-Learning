class Node:
    def __init__(self):
        self.attribute = None
        self.children = {}
        self.label = None

    # you may want to add additional fields here...

    def __init__(self, attribute, children, label):
        self.attribute = attribute
        self.children = children
        self.label = label

    def get_attribute(self):
        return self.attribute

    def get_child(self, value):
        return self.children[value]

    def get_children(self):
        return self.children

    def get_label(self):
        return self.label

    def add_child(self, child_node):
        self.children[child_node.get_attribute()] = child_node
