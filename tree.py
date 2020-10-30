'''
Tree data structure for posterior matching scheme
 - Splay tree
 - AVL tree (NA)
'''

class Tree():
    def __init__(self, start_value, length, prob):
        self.start_value, self.length, self.p = start_value, length, prob
        self.parent, self.left, self.right = None, None, None 

class SplayTree(Tree):
    def __init__(self, start_value, length, prob):
        super().__init__(start_value, length, prob)

    def insert(self, node):
        node.parent = self
        self.left = node
        self.right = SplayTree(node.start_value + node.length, self.length - node.length, 1 - node.p)
        self.right.parent = self
    
    def search_node(self, prob, direction):
        # direction = 0 or 1: fulfill left part or right part first
        p = prob
        if not self.left: # leaf
            if direction == 0:
                new_node = SplayTree(self.start_value, self.length * p, p)
                self.insert(new_node)
            else:
                new_node = SplayTree(self.start_value, self.length * (1-p), 1-p)
                self.insert(new_node)
            return self.right
        else:
            if direction == 0:
                if self.left.p < p: # the left child's CDF is not enough
                    return self.right.search_node( (p - self.left.p) / self.right.p, direction)
                else:   # find the midddle in left
                    return self.left.search_node( p / self.left.p, direction )
            else:
                if self.right.p < p:
                    return self.left.search_node( (p - self.right.p) / self.left.p, direction)
                else:
                    return self.right.search_node( p / self.right.p, direction )

    def zig(self): # right rotation
        # update probability
        self.left.p *= self.p
        self.right.p *= self.p
        self.p = self.parent.p
        self.parent.p = 1 - self.left.p
        self.right.p /= self.parent.p
        self.parent.right.p = 1 - self.right.p

        # update value and length
        self.start_value = self.parent.start_value # maybe optional
        self.length = self.parent.length
        self.parent.start_value = self.right.start_value
        self.parent.length -= self.left.length
        
        grandparent = self.parent.parent
        # connect right child with parent, disconnect right child
        self.parent.left = self.right
        self.right.parent = self.parent
        self.right = self.parent
        # disconnect parent and grandparent, re-connect its parent
        self.right.parent = self
        self.parent = grandparent
        if grandparent and grandparent.left is self.right:
            grandparent.left = self
        elif grandparent and grandparent.right is self.right:
            grandparent.right = self
        elif not grandparent:
            pass
        else:
            print("Error! Grandparent and parent are not matched in zig!\n {}\n {}\n {}".format(grandparent.start_value, self.left.start_value, self.right.start_value))
            exit()
        return self

    def zag(self): # left rotation
        # update probability
        self.left.p *= self.p
        self.right.p *= self.p
        self.p = self.parent.p
        self.parent.p = 1 - self.right.p
        self.left.p /= self.parent.p
        self.parent.left.p = 1 - self.left.p

        # update value and length
        self.start_value = self.parent.start_value
        self.length = self.parent.length
        self.parent.length -= self.right.length

        grandparent = self.parent.parent
        # connect right child with parent, disconnect left child
        self.parent.right = self.left
        self.left.parent = self.parent
        self.left = self.parent
        # disconnect parent and grandparent, re-connect its parent
        self.left.parent = self
        self.parent = grandparent
        if grandparent and grandparent.left is self.left:
            grandparent.left = self
        elif grandparent and grandparent.right is self.left:
            grandparent.right = self
        elif not grandparent:
            pass
        else:
            print("Error! Grandparent and parent are not matched in zag!\n {}\n {}\n {}".format(grandparent.start_value, self.left.start_value, self.right.start_value))
            exit()
        return self
    
    def rotate(self, subtree=False):
        if not self.parent:
            return self
        if not self.parent.parent:
            if subtree: # root of subtree
                return self
            return self.zig() if self.parent.left is self else self.zag()
        
        grandparent = self.parent.parent
        # grandparent, parent and child are on the same side
        # zig-zig
        if grandparent.left is self.parent and self.parent.left is self:
            return self.parent.zig().left.zig().rotate()
        # zag-zag
        elif grandparent.right is self.parent and self.parent.right is self:
            return self.parent.zag().right.zag().rotate()
        # grandparent, parent and child are on the diff sides
        elif grandparent.left is self.parent and self.parent.right is self:
            return self.zag().zig().rotate()
        elif grandparent.right is self.parent and self.parent.left is self:
            return self.zig().zag().rotate()
        else:
            print("Error! No correction pattern!")
            exit()

    def print_node(self, text='NODE'):
        """ Print out info about node, its children and its parent """
        print('--------------- {} ---------------'.format(text))
        n = self.parent
        if n is not None:
            print('Root: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        else:
            print('Root: {}'.format(n))

        n = self
        if n is not None:
            print('Tree node: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        
        n = self.left
        if n is not None: 
            print('Left child: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        else:
            print('Left child: {}'.format(n))
        
        n = self.right
        if n is not None:
            print('Right child: v={} l={} p={}'.format(n.start_value, n.length, n.p))
        else:
            print('Right child: {}'.format(n))
        print()



class AVLTree(Tree): # TODO
    def __init__(self, start_value, length, prob):
        super().__init__(start_value, length, prob) # TODO