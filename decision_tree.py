# Codes modified from: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
class decision_tree(object):
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
    
    def _gini_index(self, groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size==0: continue
            score = 0.0
            for class_val in classes:
                p = [example[-1] for example in group].count(class_val)/size
                score += p*p
                gini += (1.0 - score)*(size/n_instances)
        return gini

    def _terminated(self, group):
        outcomes = [example[-1] for example in group]
        return max(set(outcomes), key=outcomes.count)

    def _test_split(self, feature, value, dataset):
        left, right = [], []
        for example in dataset:
            if example[feature] < value: left.append(example)
            else: right.append(example)
        return left, right

    def _get_split(self, data):
        class_values = list(set(row[-1] for row in data))
        base_feature, base_value, base_score, base_groups = 999, 999, 999, None
        for feature in range(len(data[0])-1):
            for example in data:
                groups = self._test_split(feature, example[feature], data)
                gini = self._gini_index(groups, class_values)
                if gini<base_score:
                    base_feature, base_value, base_score, base_groups = feature, example[feature], gini, groups
        return {'feature': base_feature, 'value':base_value, 'groups':base_groups}

    def _split(self, root, depth):
        left, right = root['groups']
        del(root['groups'])

        if not left or not right:
            root['left'] = root['right'] = self._terminated(left + right)
            return
        
        if depth >= self.max_depth:
            root['left'], root['right'] = self._terminated(left), self._terminated(right)
            return

        if len(left)<=self.min_size:
            root['left'] = self._terminated(left)
        else:
            root['left'] = self._get_split(left)
            self._split(root['left'], depth+1)

        if len(right) <= self.min_size:
            root['right'] = self._terminated(right)
        else: 
            root['right'] = self._get_split(right)
            self._split(root['right'], depth+1)

    def fit(self, train_data):
        root = self._get_split(train_data)
        self._split(root, 1)
        self.root = root

    def predict(self, test_data):
        def predict(root, test_data):
            if test_data[root['feature']]<root['value']:
                if isinstance(root['left'], dict):
                    return predict(root['left'], test_data)
                else: return root['left']
            else:
                if isinstance(root['right'], dict):
                    return predict(root['right'], test_data)
                else: return root['right']
        return predict(self.root, test_data)