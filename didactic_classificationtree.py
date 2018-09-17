import itertools
import pydotplus
import numpy as np

from IPython.display import Image, display
from collections import defaultdict
from fractions import Fraction
from sklearn.metrics import confusion_matrix


def print_fraction(f, denominator):
    if f.denominator == denominator:
        return f
    else:
        k = denominator/f.denominator
        return '%s/%s' % (f.numerator * k, denominator)


def get_color(distribution):
    ndistr = [1.0 * v / sum(distribution) for v in distribution]
    maxval = max(ndistr)
    argmaxval = np.argmax(ndistr)
    color = '#f7f7f7'
    if maxval <= 0.5:
        color = '#f7f7f7'
    elif 0.5 < maxval <= 0.75:
        color = '#c2a5cf' if argmaxval < len(ndistr) / 2.0 else '#a6dba0'
    elif maxval > 0.75:
        color = '#7b3294' if argmaxval < len(ndistr) / 2.0 else '#008837'

    return color


def error_rate(p, d=None):
    ret = Fraction(1, 1) - np.max(p)
    v1 = np.max(p) if d is None else print_fraction(np.max(p), d)
    v2 = ret if d is None else print_fraction(ret, d)
    print('\t%s - %s = %s' % (Fraction(1, 1), v1, v2))
    return ret


def gini(p, d=None):
    ret = Fraction(1, 1)
    val = 0
    val_str = ''
    for n in p:
        val += n**2
        val_str += '(%d/%d)^2 + ' % (n.numerator, n.denominator)
    val_str = val_str[:-3]
    ret -= val
    print('\t%s - (%s) = %s' % (Fraction(1, 1), val_str, ret))
    return ret


class DidatticClassificationTree:
    def __init__(self, fun=error_rate, fun_name='misc err', min_samples_split=2, min_samples_leaf=1,
                 step_by_step=False):

        self.fun = fun
        self.fun_name = fun_name
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.step_by_step = step_by_step
        self.jdata = None

    def fit(self, dataset_df, target, plot_figures=True):
        self.target = target
        self.jdata = dict()
        self.jdata['train'] = dataset_df.values.tolist()
        self.jdata['target'] = [target, dataset_df.columns.tolist().index(target)]
        self.jdata['parameters'] = {
            'split_function': self.fun_name,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
        }

        treestr, _, _, node = self.iteration(dataset_df, target, self.fun, self.fun_name)
        self.jdata['tree'] = node

        if self.step_by_step:
            val = input('')

        dot_str = """digraph Tree {
node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
%s}""" % treestr

        tredotfile = open('tree.dot', 'w')
        tredotfile.write(dot_str)
        tredotfile.close()

        if plot_figures:
            with open('tree.dot', 'r') as dot_data:
                dot_data = dot_data.read()
            graph = pydotplus.graph_from_dot_data(dot_data)
            display(Image(graph.create_png()))

        self.tree_nodes, self.tree_leaves, self.tree_edges = self.build_classifier(treestr)

    def predict(self, test_df):
        predicted_values = list()
        for rowid in range(0, len(test_df)):
            idnode = '0'
            pred = self.make_prediction(idnode, rowid, test_df, self.tree_nodes, self.tree_leaves, self.tree_edges)
            predicted_values.append(pred)

        return predicted_values

    def evaluate(self, test_df, predicted='Predicted', plot_figures=True):
        self.jdata['test'] = test_df.values.tolist()
        self.jdata['evaluation'] = dict()
        val = sorted(np.unique(test_df[self.target].values))
        tp, tn, fp, fn = self.get_confusion_matrix(test_df, self.target, predicted)
        self.jdata['evaluation']['confusion_matrix'] = [tp, tn, fp, fn]

        if plot_figures:
            self.print_confusion_matrix(val, tp, tn, fp, fn)

        p = self.precision(tp, tn, fp, fn)
        r = self.recall(tp, tn, fp, fn)
        a = self.accuracy(tp, tn, fp, fn)
        f1 = self.fmeasure(tp, tn, fp, fn)

        if plot_figures:
            print('Precision', p, float(p))
            print('Recall', r, float(r))
            print('F1-measure', f1, float(f1))
            print('Accuracy', a, float(a))
        self.jdata['evaluation']['precision'] = [str(p), float(p)]
        self.jdata['evaluation']['recall'] = [str(r), float(r)]
        self.jdata['evaluation']['f1score'] = [str(f1), float(f1)]
        self.jdata['evaluation']['accuracy'] = [str(a), float(a)]

    def get_confusion_matrix(self, test_df, target, predicted='Predicted'):
        cm = confusion_matrix(test_df[target], test_df[predicted])

        tp = int(cm[1][1])
        tn = int(cm[0][0])
        fp = int(cm[0][1])
        fn = int(cm[1][0])

        return tp, tn, fp, fn

    def print_confusion_matrix(self, val, tp, tn, fp, fn):
        print('R\P\t|%s\t|%s\t|' % (val[0], val[1]))
        print('%s\t|%d\t|%d\t|' % (val[0], tn, fp))
        print('%s\t|%d\t|%d\t|' % (val[1], fn, tp))

    def precision(self, tp, tn, fp, fn):
        return Fraction(tp, tp + fp)

    def recall(self, tp, tn, fp, fn):
        return Fraction(tp, tp + fn)

    def accuracy(self, tp, tn, fp, fn):
        return Fraction(tp + tn, tp + tn + fp + fn)

    def fmeasure(self, tp, tn, fp, fn):
        p = self.precision(tp, tn, fp, fn)
        r = self.recall(tp, tn, fp, fn)
        return 2 * p * r / (p + r)

    def get_parent_gain(self, dataset_df, target, fun):
        parent_dict = defaultdict(int)
        for row in dataset_df[[target]].values:
            parent_dict[tuple(row)] += 1
        total = len(dataset_df)

        p = list()
        for k, v in parent_dict.items():
            p.append(Fraction(v, total))
        return fun(p, len(dataset_df))

    def get_children_gain(self, dataset_df, feature, target, fun):

        calculus = ''
        count_dict = defaultdict(int)
        weight_dict = defaultdict(int)
        for row in dataset_df[[feature, target]].values:
            count_dict[tuple(row)] += 1
            weight_dict[row[0]] += 1

        print(feature, sorted(weight_dict.keys()))
        calculus += feature + ','.join(sorted(weight_dict.keys()))

        total = len(dataset_df)

        splitvalue_gain = dict()
        for sv in sorted(weight_dict):
            print('\t', sv, weight_dict[sv])
            calculus += '\t %s %s' % (sv, weight_dict[sv])
            if weight_dict[sv] > 0:
                p = list()
                for k in sorted(count_dict):
                    v = count_dict[k]
                    if k[0] == sv:
                        print('\t\t%s, %d/%d' % (k[1], count_dict[k], weight_dict[sv]))
                        calculus += '\t\t%s, %d/%d' % (k[1], count_dict[k], weight_dict[sv])
                        p.append(Fraction(v, weight_dict[sv]))
                print('\t', end=',')
                calculus += '\t'
                splitvalue_gain[sv] = fun(p, weight_dict[sv])
            else:
                splitvalue_gain[sv] = Fraction(0, 1)

        gain_child = 0
        print_str = ''
        for sv in splitvalue_gain:
            v1 = print_fraction(splitvalue_gain[sv], weight_dict[sv])
            v2 = print_fraction(Fraction(weight_dict[sv], total), total)
            print_str += '(%s * %s) + ' % (v1, v2)
            gain_child += splitvalue_gain[sv] * Fraction(weight_dict[sv], total)

        print_str = print_str[:-2] + '= %s' % print_fraction(gain_child, total)
        print('\t', print_str)
        calculus += '\t %s' % print_str
        return gain_child, calculus

    def calculate_children_gains(self, dataset_df, target, fun, parentgain):
        feature_childgain = dict()
        feature_childgain_calculus = ''
        for feature, ftype in dataset_df.dtypes.items():
            if feature == target:
                continue

            values = dataset_df[feature].unique()

            if ftype == object and len(values) == 2:
                cg, calculus = self.get_children_gain(dataset_df, feature, target, fun)
                feature_childgain_calculus += calculus
                key = feature + '#' + '&'.join(sorted(values))
                feature_childgain[key] = (parentgain - cg, ftype, len(values))
                v1 = print_fraction(parentgain, len(dataset_df))
                v2 = print_fraction(cg, len(dataset_df))
                v3 = print_fraction(parentgain - cg, len(dataset_df))
                print('\tDelta Gain: %s - %s = %s\n' % (v1, v2, v3))
                feature_childgain_calculus += '\tDelta Gain: %s - %s = %s\n' % (v1, v2, v3)
                if self.step_by_step:
                    val = input('')

            elif ftype == object and len(values) > 2:
                cg, calculus = self.get_children_gain(dataset_df, feature, target, fun)
                feature_childgain_calculus += calculus
                key = feature + '#' + '&'.join(sorted(values))
                feature_childgain[key] = (cg, ftype, len(values))
                feature_tmp = feature + '#'
                for nvalues in range(2, len(values)):
                    for splits in itertools.combinations(values, nvalues):
                        feature1 = '-'.join(set(splits))
                        feature2 = '-'.join(set(values) - set(splits))
                        dataset_df[feature_tmp] = np.where(dataset_df[feature].isin(set(splits)), feature1, feature2)
                        cg, calculus = self.get_children_gain(dataset_df, feature_tmp, target, fun)
                        feature_childgain_calculus += calculus
                        key = feature_tmp + feature1 + '&' + feature2
                        feature_childgain[key] = (parentgain - cg, ftype, 2)
                        dataset_df = dataset_df.drop(feature_tmp, 1)
                        v1 = print_fraction(parentgain, len(dataset_df))
                        v2 = print_fraction(cg, len(dataset_df))
                        v3 = print_fraction(parentgain - cg, len(dataset_df))
                        print('\tDelta Gain: %s - %s = %s\n' % (v1, v2, v3))
                        feature_childgain_calculus += '\tDelta Gain: %s - %s = %s\n' % (v1, v2, v3)
                        if self.step_by_step:
                            val = input('')

            elif ftype == int:
                feature_tmp = feature + '#'
                sorted_values = sorted(values)
                for i in range(0, len(sorted_values) - 1):
                    thr = sorted_values[i]
                    feature1 = '<=%s' % thr
                    feature2 = '>%s' % thr
                    dataset_df[feature_tmp] = np.where(dataset_df[feature] <= thr, feature1, feature2)
                    cg, calculus = self.get_children_gain(dataset_df, feature_tmp, target, fun)
                    feature_childgain_calculus += calculus
                    key = feature_tmp + str(thr) 
                    feature_childgain[key] = (parentgain - cg, ftype, 2)
                    dataset_df = dataset_df.drop(feature_tmp, 1)
                    v1 = print_fraction(parentgain, len(dataset_df))
                    v2 = print_fraction(cg, len(dataset_df))
                    v3 = print_fraction(parentgain - cg, len(dataset_df))
                    print('\tDelta Gain: %s - %s = %s\n' % (v1, v2, v3))
                    feature_childgain_calculus += '\tDelta Gain: %s - %s = %s\n' % (v1, v2, v3)
                    if self.step_by_step:
                        val = input('')

        return feature_childgain, feature_childgain_calculus

    def select_best_gain(self, feature_childgain, nrecords):
        maxgain = Fraction(0, 1)
        maxfeature = None
        maxftype = None
        maxsplitway = None
        for feature in feature_childgain:
            delta_gain, ftype, splitway = feature_childgain[feature]
            # dgstr = print_fraction(delta_gain, nrecords)
            if delta_gain > 0 and delta_gain > maxgain:
                maxgain = delta_gain
                maxfeature = feature
                maxftype = ftype
                maxsplitway = splitway
            elif delta_gain > 0 and delta_gain == maxgain and ftype != maxftype and ftype == object:
                maxgain = delta_gain
                maxfeature = feature
                maxftype = ftype
                maxsplitway = splitway
            elif delta_gain > 0 and delta_gain == maxgain and ftype == maxftype and splitway > maxsplitway:
                maxgain = delta_gain
                maxfeature = feature
                maxftype = ftype
                maxsplitway = splitway

        return maxfeature, maxgain, maxftype, maxsplitway

    def get_subdataset(self, dataset_df, target, maxfeature, maxgain, maxftype, maxsplitway):
        featuresplit = maxfeature.split('#')[0]
        values = maxfeature.split('#')[1]

        split_indexes = defaultdict(list)
        remove_featuresplit = False
        if maxftype == object and '-' not in values:
            remove_featuresplit = True
            for i, row in enumerate(dataset_df[[featuresplit, target]].values):
                split_indexes[row[0]].append(i)
        elif maxftype == object and '-' in values:
            p1 = values.split('&')[0]
            p2 = values.split('&')[1]
            v1 = set(p1.split('-'))
            v2 = set(p2.split('-'))
            for i, row in enumerate(dataset_df[[featuresplit, target]].values):
                if row[0] in v1:
                    split_indexes[p1].append(i)
                elif row[0] in v2:
                    split_indexes[p2].append(i)
        elif maxftype == int:
            thr = float(values)
            for i, row in enumerate(dataset_df[[featuresplit, target]].values):
                if row[0] <= thr:
                    split_indexes['<=%s' % thr].append(i)
                elif row[0] > thr:
                    split_indexes['>%s' % thr].append(i)

        feature_subdataset = dict()
        for feature in split_indexes:
            subdataset_df = dataset_df.iloc[split_indexes[feature]].copy(deep=True)
            if remove_featuresplit:
                subdataset_df = subdataset_df.drop(featuresplit, 1)
            if len(subdataset_df) < self.min_samples_leaf:
                return None
            feature_subdataset[feature] = subdataset_df

        return feature_subdataset

    def get_leaf(self, dataset_df, target, fun_name, parentgain, idnode, final_target_values=['NO', 'YES']):
        nsamples = len(dataset_df)
        distribution_dict = defaultdict(int)
        for row in dataset_df[target].values:
            distribution_dict[row] += 1
        classmajority = None
        classmajority_val = 0
        for target_val in sorted(distribution_dict):
            if classmajority_val < distribution_dict[target_val]:
                classmajority_val = distribution_dict[target_val]
                classmajority = target_val
        distribution = [distribution_dict[k] for k in sorted(distribution_dict)]

        if len(distribution) > 1:
            color = get_color(distribution)
        else:
            if list(distribution_dict.keys())[0] == final_target_values[0]:
                color = '#7b3294'
                distribution.append(0)
            elif list(distribution_dict.keys())[0] == final_target_values[1]:
                color = '#008837'
                distribution.insert(0, 0)
            else:
                color = '#f7f7f7'

        leaf = """%s [label=<%s = %.2f<br/>samples = %s<br/>value = %s<br/>class = %s>, fillcolor="%s"] ;\n""" \
               % (idnode, fun_name, parentgain, nsamples, distribution, classmajority, color)

        return leaf, classmajority

    def get_node(self, dataset_df, target, fun_name, feature_subdataset, maxfeature, maxgain, maxftype, idnode):
        feature_pdot = maxfeature.split('#')[0]
        operation = '=' if maxftype == object else '&le;'
        value = sorted(feature_subdataset.keys())[0].replace('&', '-')
        if maxftype != object:
            value = str(list(set(value.replace('<=', '').replace('>', '').split(',')))[0])
        function = fun_name
        gain = maxgain
        nsamples = len(dataset_df)

        distribution = list()
        classmajority = None
        classmajority_val = 0

        distribution_dict = defaultdict(int)
        for row in dataset_df[target].values:
            distribution_dict[row] += 1
        classmajority = None
        classmajority_val = 0
        for target_val in sorted(distribution_dict):
            if classmajority_val < distribution_dict[target_val]:
                classmajority_val = distribution_dict[target_val]
                classmajority = target_val
        distribution = [distribution_dict[k] for k in sorted(distribution_dict)]

        color = get_color(distribution)
        node = """%s [label=<%s %s %s?<br/>%s = %.2f<br/>samples = %s<br/>value = %s>, fillcolor="%s"] ;\n""" \
               % (idnode, feature_pdot, operation, value, function, gain, nsamples, distribution, color)

        return node

    def iteration(self, dataset_df, target, fun, fun_name, pathfeature=None, idnode=0):

        # self.jdata['tree']['node']
        values_json = dataset_df[target].value_counts().to_dict()
        values_json = {k: int(v) for k, v in values_json.items()}
        jnode = {
            'name': '',
            'parent_gain': 0.0,
            # 'split_by': None,
            # 'split_gain': None,
            'records': len(dataset_df),
            'values': values_json,
            # 'class_majority': None,
            'children': list(),
            'calculus': ''
        }

        if len(dataset_df) < self.min_samples_leaf:
            leaf, classmajority = self.get_leaf(dataset_df, target, fun_name, 0.0, idnode)
            print('--> Min leaf constraint. Stop\n')
            jnode['calculus'] += '--> Min leaf constraint. Stop\n'
            jnode['class_majority'] = classmajority
            if self.step_by_step:
                val = input('')
            return leaf, idnode, idnode, jnode

        nfeatures = len(dataset_df.columns)
        if pathfeature is None:
            pathfeature = 'Root'

        print(pathfeature)
        jnode['calculus'] += pathfeature
        jnode['name'] = pathfeature
        print('Parent')
        jnode['calculus'] += 'Parent'
        parentgain = self.get_parent_gain(dataset_df, target, fun)
        jnode['parent_gain'] = float(parentgain)
        if parentgain == 0:
            leaf, classmajority = self.get_leaf(dataset_df, target, fun_name, parentgain, idnode)
            print('--> No gain 1. Stop\n')
            jnode['calculus'] += '--> No gain. Stop\n'
            jnode['class_majority'] = classmajority
            if self.step_by_step:
                val = input('')
            return leaf, idnode, idnode, jnode

        feature_childgain, feature_childgain_str = self.calculate_children_gains(dataset_df, target, fun, parentgain)
        jnode['calculus'] += feature_childgain_str
        jnode['children_gain'] = {k: float(v[0]) for k, v in feature_childgain.items()}

        dataset_df = dataset_df[dataset_df.columns[:nfeatures]]
        maxfeature, maxgain, maxftype, maxsplitway = self.select_best_gain(feature_childgain, len(dataset_df))
        if maxfeature is not None:
            jnode['split_by'] = maxfeature
            jnode['split_gain'] = float(maxgain)
            print('--> Split By', maxfeature, '\n')
            jnode['calculus'] += '--> Split By %s\n' % maxfeature
            feature_subdataset = self.get_subdataset(dataset_df, target, maxfeature,
                                                     maxgain, maxftype, maxsplitway)

            if feature_subdataset is not None:

                node = self.get_node(dataset_df, target, fun_name, feature_subdataset,
                                     maxfeature, maxgain, maxftype, idnode)

                nodestr = ''
                nodestr += node

                idnodeparent = idnode
                leftright = 'left'
                idnodemax = idnode
                for feature in sorted(feature_subdataset):
                    subdataset_df = feature_subdataset[feature]
                    subnode, idnode, idnodemax, jnode_children = self.iteration(subdataset_df, target, fun, fun_name,
                                                                pathfeature + '_' + maxfeature + '?' + feature,
                                                                idnodemax + 1)
                    jnode['children'].append(jnode_children)
                    nodestr += subnode
                    if leftright == 'left':
                        nodestr += """%s -> %s [labeldistance=2.5, labelangle=90, headlabel="%s"] ;\n""" \
                                   % (idnodeparent, idnode, feature)
                        leftright = 'right'
                    else:
                        nodestr += """%s -> %s [labeldistance=2.5, labelangle=-90, headlabel="%s"] ;\n""" \
                                   % (idnodeparent, idnode, feature)

                return nodestr, idnodeparent, idnodemax, jnode

            else:
                leaf, classmajority = self.get_leaf(dataset_df, target, fun_name, parentgain, idnode)
                print('--> Min leaf constraint 2. Stop\n')
                jnode['calculus'] += '--> Min leaf constraint. Stop\n'
                jnode['class_majority'] = classmajority
                if self.step_by_step:
                    val = input('')
                return leaf, idnode, idnode, jnode
        else:
            leaf, classmajority = self.get_leaf(dataset_df, target, fun_name, parentgain, idnode)
            print('--> No gain 2. Stop\n')
            jnode['calculus'] += '--> No gain. Stop\n'
            jnode['class_majority'] = classmajority
            if self.step_by_step:
                val = input('')
            return leaf, idnode, idnode, jnode

    def build_classifier(self, treestr):
        tree_nodes = dict()
        tree_leaves = dict()
        tree_edges = defaultdict(list)

        for row in treestr.split('\n'):
            if len(row) == 0:
                continue

            idnode = row.split(' ')[0]
            if row.split(' ')[1] == '->':
                childnode = row.split(' ')[2]
                label = row.split(' ')[5][10:].replace('"', '').replace(']', '')
                tree_edges[idnode].append([childnode, label])
            else:
                classindex = row.find('class =')
                if classindex > 0:
                    prediction = row[classindex + len('class = '):row.find('>, fillcolor=')]
                    tree_leaves[idnode] = prediction
                else:
                    feature_op_value = row[row.find('[label=<') + len('[label=<'):row.find('?<br/>')]
                    if '=' in feature_op_value:
                        feature = feature_op_value.split(' = ')[0]
                        value = feature_op_value.split(' = ')[1]
                        op = '='
                        tree_nodes[idnode] = {'feature': feature, 'op': op, 'value': value}
                    else:
                        feature = feature_op_value.split(' &le; ')[0]
                        value = feature_op_value.split(' &le; ')[1]
                        op = '&le;'
                        tree_nodes[idnode] = {'feature': feature, 'op': op, 'value': value}

        return tree_nodes, tree_leaves, tree_edges

    def make_prediction(self, idnode, rowid, test_df, tree_nodes, tree_leaves, tree_edges):
        if idnode in tree_nodes:
            feature = tree_nodes[idnode]['feature']
            op = tree_nodes[idnode]['op']
            value = tree_nodes[idnode]['value']
            record_value = test_df.ix[rowid][feature]
            if op == '=':
                if record_value in value.split('-'):    
                    for nextnode_val in tree_edges[idnode]:
                        if record_value in nextnode_val[1].split('-'):
                            idnode = nextnode_val[0]
                            break
                    return self.make_prediction(idnode, rowid, test_df, tree_nodes, tree_leaves, tree_edges)
                else:
                    for nextnode_val in tree_edges[idnode]:
                        if record_value in nextnode_val[1].split('-'):
                            idnode = nextnode_val[0]
                            break
                    return self.make_prediction(idnode, rowid, test_df, tree_nodes, tree_leaves, tree_edges)
            else:
                record_value = float(record_value)
                value = float(value)
                if record_value <= value:
                    for nextnode_val in tree_edges[idnode]:
                        if nextnode_val[1] == ('<=%s' % value):
                            idnode = nextnode_val[0]
                            break
                    return self.make_prediction(idnode, rowid, test_df, tree_nodes, tree_leaves, tree_edges)
                else:
                    for nextnode_val in tree_edges[idnode]:
                        if nextnode_val[1] == ('>%s' % value):
                            idnode = nextnode_val[0]
                            break
                    return self.make_prediction(idnode, rowid, test_df, tree_nodes, tree_leaves, tree_edges)

        if idnode in tree_leaves:
            return tree_leaves[idnode]

    def get_jdata(self):
        return self.jdata




