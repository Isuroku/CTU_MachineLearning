import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from collections import deque
from IPython.display import Image
from sklearn.model_selection import train_test_split
from time import time
import threading
import multiprocessing as mp


class Dataset(object):
    """
    This class is a representation of a (subset of) dataset optimized for splitting needed for construction of
    regression trees. Actual data are not copied, indices are kept, only.
    """

    def __init__(self, df, ix=None):
        """
        Constructor
        :param df: Pandas DataFrame or another Dataset instance. In the latter case only meta data are copied.
        :param ix: boolean index describing samples selected from the original dataset
        """
        if isinstance(df, pd.DataFrame):
            self.columns = list(df.columns)
            self.cdict = {c: i for i, c in enumerate(df.columns)}
            self.data = [df[c].values for c in self.columns]
        elif isinstance(df, Dataset):
            self.columns = df.columns
            self.cdict = df.cdict
            self.data = df.data
            assert ix is not None
        self.ix = np.arange(len(self.data[0]), dtype=np.int64) if ix is None else ix

    def __getitem__(self, cname):
        """
        Returns dataset column.
        :param cname: column name
        :return: the column as numpy array
        """
        return self.data[self.cdict[cname]][self.ix]

    def get_data(self, cname):
        c = self.cdict[cname]
        values = self.data[c]
        res = np.zeros((2, len(self.ix)))
        res[0] = self.ix
        for i in range(len(self.ix)):
            index = int(res[0, i])
            t = values[index]
            res[1, i] = t
        return res

    def __len__(self):
        """
        The number of samples
        :return:
        """
        return len(self.ix)

    def get_bootstrap_indices(self, n_samples):
        indices = np.random.randint(0, len(self.ix), n_samples)
        indices = np.sort(indices)
        return indices

    def to_dict(self):
        """
        Return the data in a form used in prediction.
        :return: list of dicts with dict for each data sample, keys are the column names
        """
        return [{c: self.data[self.cdict[c]][i] for c in self.columns} for i in self.ix]

    def modify_col(self, cname, d):
        """
        Creates a copy of this dataset replacing one of its columns data. This method might be helpful for the Gradient
        Boosted Trees.
        :param cname: column name
        :param d: a numpy aray with new column data
        :return: new Dataset
        """
        assert len(self.ix) == len(self.data[0]), 'works for unfiltered rows, only'
        new_dataset = Dataset(self, ix=self.ix)
        new_dataset.data = list(self.data)
        new_dataset.data[self.cdict[cname]] = d
        return new_dataset

    def filter_rows(self, cname, cond):
        """
        Creates a new Dataset containing only the rows satisfying a given condition.
        :param cname: column name
        :param cond: condition
        :return:
        """
        col = self[cname]
        return Dataset(self, ix=self.ix[cond(col)])


def generate_sin_data(n, random_x=False, scale=0.0):
    """
    Sin dataset generator.
    """
    rng = np.random.RandomState(1234)
    if random_x:
        X = rng.uniform(0, 2 * np.pi, n)
    else:
        X = np.linspace(0, 2 * np.pi, n)
    T = np.sin(X) + rng.normal(0, scale, size=X.shape)
    df = pd.DataFrame({'x': X, 't': T}, columns=['x', 't'])
    return Dataset(df), np.sqrt(np.mean((T - np.sin(X)) ** 2))


def generate_boston_housing():
    """
    Import Boston housing.
    """
    # https://www.kaggle.com/c/boston-housing/data
    df = pd.read_csv('housing.csv')
    df.drop(['ID'], axis=1, inplace=True)  # remove unwanted column
    data_housing_train, data_housing_test = train_test_split(df, test_size=0.3, random_state=1)
    return Dataset(data_housing_train), Dataset(data_housing_test)


class DecisionNode(object):
    """
    Represents an inner decision node
    """

    def __init__(self, attr, value, left, right):
        """
        Constructs a node
        :param attr: splitting attribute
        :param value: splitting attribute value
        :param left: left child
        :param right: right child
        """
        self.attr = attr
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        """
        Evaluates the node.
        :param x: a dictionary: key = attribute (column) name, value = attribute value
        :return: reference to the corresponding child
        """
        if isinstance(self.value, str):
            if x[self.attr] == self.value:
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)
        else:
            if x[self.attr] <= self.value:
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)

    def get_nodes(self):
        """
        Return all nodes of the subtree rooted in this node.
        """
        ns = []
        q = deque([self])
        while len(q) > 0:
            n = q.popleft()
            ns.append(n)
            if isinstance(n, DecisionNode):
                q.append(n.left)
                q.append(n.right)
        return ns

    def __str__(self):
        """
        String representation f
        :return:
        """
        if isinstance(self.value, str):
            return '{}=="{}"'.format(self.attr, self.value)
        else:
            return '{}<={:5.2f}'.format(self.attr, self.value)


class LeafNode(object):
    def __init__(self, response):
        self.response = response

    def evaluate(self, x):
        return self.response

    def get_nodes(self):
        return [self]

    def __str__(self):
        return '{:5.2f}'.format(self.response)


class RegressionTree(object):
    def __init__(self, data, tattr, xattrs=None, max_depth=5,
                 max_features=lambda n: n,
                 rng=np.random.RandomState(1)):
        """
        Regression tree constructor. Constructs the model fitting supplied dataset.
        :param data: Dataset instance
        :param tattr: the name of target attribute column
        :param xattrs: list of names of the input attribute columns
        :param max_depth: limit on tree depth
        :param max_features: the number of features considered when splitting a node (all by default)
        :param rng: random number generator used for sampling features when selecting a split candidate
        """
        self.xattrs = [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        self.tattr = tattr
        self.max_features = int(np.ceil(max_features(len(self.xattrs))))
        self.rng = rng
        self.root = self.build_tree(data, self.impurity(data), max_depth=max_depth)

    def evaluate(self, x):
        """
        Evaluates the node.
        :param x: a dictionary: key = attribute (column) name, value = attribute value
        :return: reference to the corresponding child
        """
        return self.root.evaluate(x)

    def impurity(self, data):
        """
        Impurity/loss for constant mean model of data. Squared loss used here.
        """
        if len(data) == 0:
            return 0.0
        t = data[self.tattr]
        return np.sum((t - t.mean()) ** 2)

    def build_tree(self, data, impurity, max_depth):
        if max_depth > 0:
            # if len(data) > 5:
            best_impurity = impurity
            xattrs = self.rng.choice(self.xattrs, self.max_features,
                                     replace=False)  # select attributes to be considered
            for xattr in xattrs:
                vals = np.unique(data[xattr])  # get all unique values of the attribute
                if len(vals) <= 1: continue
                for val in vals:
                    if isinstance(val, str):
                        data_l = data.filter_rows(xattr, lambda a: a == val)
                        data_r = data.filter_rows(xattr, lambda a: a != val)
                    else:
                        data_l = data.filter_rows(xattr, lambda a: a <= val)
                        data_r = data.filter_rows(xattr, lambda a: a > val)

                    impurity_l = self.impurity(data_l)
                    impurity_r = self.impurity(data_r)

                    split_impurity = impurity_l + impurity_r  # total impurity if splitting

                    if split_impurity < best_impurity and len(data_l) > 0 and len(data_r) > 0:
                        best_impurity, best_xattr, best_val = split_impurity, xattr, val
                        best_data_l, best_data_r = data_l, data_r
                        best_impurity_l, best_impurity_r = impurity_l, impurity_r

            if best_impurity < impurity:  # splitting reduces the impurity, choose best possible split
                return DecisionNode(best_xattr, best_val,
                                    self.build_tree(best_data_l, best_impurity_l, max_depth - 1),
                                    self.build_tree(best_data_r, best_impurity_r, max_depth - 1))
        return LeafNode(data[self.tattr].mean())

    def plot(self):
        """
        Plots trees. Useful for debugging. You have to install networkx and pydot Python modules as well as graphviz.
        Display in Jupyter notebook or save the plot to file:
            img = tree.plot()
            with open('tree.png', 'wb') as f:
            f.write(img.data)
        """
        import os
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

        import networkx as nx
        g = nx.DiGraph()
        V = self.root.get_nodes()
        d = {}
        for i, n in enumerate(V):
            d[n] = i
            g.add_node(i, label='{}'.format(n))
        for n in V:
            if isinstance(n, DecisionNode):
                g.add_edge(d[n], d[n.left])
                g.add_edge(d[n], d[n.right])

        dot = nx.drawing.nx_pydot.to_pydot(g)
        return Image(dot.create_png())


def evaluate_all(model, data):
    """
    Makes predictions for all dataset samples.
    :param model: any model implementing evaluate(x) method
    :param data: Dataset instance
    :return: predictions as a numpy array
    """
    return np.r_[[model.evaluate(x) for x in data.to_dict()]]


def rmse(model, data):
    """
    Evaluates RMSE on a dataset
    :param model: any model implementing evaluate(x) method
    :param data: Dataset instance
    :return: RMSE as a float
    """
    ys = evaluate_all(model, data)
    rmse = np.sqrt(np.mean((data[model.tattr] - ys) ** 2))
    return rmse


class RandomForest(object):
    def __init__(self, data, tattr, xattrs=None,
                 n_trees=1000,
                 max_depth=np.inf,
                 max_features=lambda n: n,
                 rng=np.random.RandomState(1),
                 multithread = False):

        self.tattr = tattr
        self.multithread = multithread

        n_samples = len(data)
        self.forest = []
        for i in range(n_trees):
            bs_indices = data.get_bootstrap_indices(n_samples)
            bs_data = Dataset(data, bs_indices)
            tree = RegressionTree(bs_data, tattr=tattr, xattrs=xattrs, max_depth=max_depth, rng=rng,
                                  max_features=max_features)
            self.forest.append(tree)

    def evaluate_onethread(self, x):
        a = np.zeros(len(self.forest))
        for i in range(len(self.forest)):
            a[i] = self.forest[i].evaluate(x)
        m = np.mean(a)
        return m

    def evaluateTree(self, tree, x, output):
        a = tree.evaluate(x)
        output.put(a)

    def evaluate_multithread(self, x):
        output = mp.Queue()
        processes = [mp.Process(target=self.evaluateTree, args=(self.forest[i], x, output)) for i in range(len(self.forest))]

        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [output.get() for p in processes]
        m = np.mean(results)

        return m

    def evaluate(self, x):
        if self.multithread:
            return self.evaluate_multithread(x)
        else:
            return self.evaluate_onethread(x)





class GradientBoostedTrees(object):
    # COMPLETE CODE HERE
    def __init__(self, data, tattr, xattrs=None,
                 n_trees=10,
                 max_depth=1,
                 beta=0.1,
                 rng=np.random.RandomState(1)):
        """
        Gradient Boosted Trees constructor. Constructs the model fitting supplied dataset.
        :param data: Dataset instance
        :param tattr: the name of target attribute column
        :param xattrs: list of names of the input attribute columns
        :param n_trees: number of trees
        :param max_depth: limit on tree depth
        :param beta: learning rate
        :param rng: random number generator
        """

        self.xattrs = [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        self.tattr = tattr
        self.max_depth = max_depth
        self.beta = beta

        self.forest = []
        self.build(data, n_trees, rng)

    @staticmethod
    def evaluate_tree(tree, data):
        return np.r_[[tree.evaluate(x) for x in data.to_dict()]]

    def evaluate_self(self, data):
        return np.r_[[self.evaluate(x) for x in data.to_dict()]]

    def build(self, data, n_trees, rng):
        tree = RegressionTree(data, tattr=self.tattr, xattrs=self.xattrs, max_depth=self.max_depth, rng=rng)
        self.forest.append(tree)

        dt = data[self.tattr]

        for k in range(n_trees):
            values = self.evaluate_self(data)
            g = dt - values
            df = data.modify_col(self.tattr, g)

            best_tree = None
            score = None
            best_attr = None
            for attr in self.xattrs:
                tmp_tree = RegressionTree(df, tattr=self.tattr, xattrs=[attr], max_depth=self.max_depth, rng=rng)
                b = self.evaluate_tree(tmp_tree, df)
                s = (g - b) ** 2
                tmp_score = np.sum(s)

                if best_tree is None:
                    best_tree = tmp_tree
                    score = tmp_score
                    best_attr = attr
                else:
                    if tmp_score < score:
                        best_tree = tmp_tree
                        score = tmp_score
                        best_attr = attr
            print('best_attr {}'.format(best_attr))
            self.forest.append(best_tree)
        pass

    def evaluate(self, x):
        res = self.forest[0].evaluate(x)
        for i in range(1, len(self.forest)):
            res += self.beta * self.forest[i].evaluate(x)
        return res


def generate_plot(ds_train, ds_test, tattr,
                  model_cls,
                  iterate_over,
                  iterate_values,
                  title,
                  xlabel,
                  rng,
                  iterate_labels=None,
                  bayes_rmse=None,
                  show_model=True,
                  show_rmse=True,
                  **model_params):
    """
    Generates plot of training and testing RMSE errors iterating over values of a selected parameter.
    :param ds_train: training Dataset instance
    :param ds_test: testing Dataset instance
    :param tattr: the name of target attribute column
    :param model_cls: model class, e.g., RegressionTree, RandomForest, GradientBoostedTrees
    :param iterate_over: the name of model parameter to iterate over (as string)
    :param iterate_values: a list of values to iterate over
    :param title: plot title
    :param xlabel: x axis label
    :param rng: random number generator
    :param iterate_labels: the labels corresponding to iterate_values, if not given, iterate_values are used instead, use for non float parameters
    :param bayes_rmse: plots the best achievable error (we know this for the sin dataset)
    :param model_params: other model parameters
    """
    train_rmses, test_rmses = [], []
    models = []
    for val in iterate_values:
        st = time()
        params = dict(model_params)
        params[iterate_over] = val
        model = model_cls(ds_train, tattr=tattr, rng=rng, **params)
        models.append(model)
        rmse_tr = rmse(model, ds_train)
        rmse_tst = rmse(model, ds_test)
        train_rmses.append(rmse_tr)
        test_rmses.append(rmse_tst)
        print('{}: {} = {} finished in {:5.2f}s. rmse_tr {}; rmse_tst {}'.format(title, iterate_over, val, time() - st,
                                                                                 rmse_tr, rmse_tst))
    best_index = np.argmin(test_rmses)

    if iterate_labels is None:
        iterate_coords = iterate_values
    else:
        assert len(iterate_labels) == len(iterate_values)
        iterate_coords = range(len(iterate_labels))

    best_model = models[best_index]
    if show_model:
        draw_model(iterate_coords, ds_train, ds_test, best_index, best_model, title)

    # img = best_model.plot()
    # with open('graph_{}.png'.format(title), 'wb') as f:
    #      f.write(img.data)

    draw_rmse(iterate_coords, train_rmses, test_rmses, bayes_rmse, best_index, xlabel, iterate_labels, title, show_rmse)


def draw_model(iterate_coords, ds_train, ds_test, best_index, best_model, title):
    y = np.sin(ds_test.data[0])
    # plt.plot(ds_test.data[0], ds_test.data[1], 'o')
    plt.plot(ds_train.data[0], ds_train.data[1], 'o')
    plt.plot(ds_test.data[0], y)

    ys = evaluate_all(best_model, ds_test)
    plt.plot(ds_test.data[0], ys)
    plt.savefig('model_{}_{}.pdf'.format(title, iterate_coords[best_index]))
    plt.show()


def draw_rmse(iterate_coords, train_rmses, test_rmses, bayes_rmse, best_index, xlabel, iterate_labels, title,
              show=True):
    plt.figure()
    plt.plot(iterate_coords, train_rmses, '.-', label='train')
    plt.plot(iterate_coords, test_rmses, '.-', label='test')
    if bayes_rmse is not None:
        plt.plot([0, iterate_coords[-1]], [bayes_rmse, bayes_rmse], '-.', label='$h^{*}(x)$')
    plt.plot(iterate_coords[best_index], test_rmses[best_index], 'o', label='best')
    plt.xlabel(xlabel)
    if iterate_labels is not None: plt.xticks(iterate_coords, iterate_labels)
    plt.ylabel('RMSE')
    plt.title('best test RMSE [{}] = {}'.format(iterate_coords[best_index], test_rmses[best_index]))
    plt.suptitle(title)
    plt.legend()
    plt.savefig('rmse_{}_{}.pdf'.format(title, iterate_coords[best_index]))
    if show:
        plt.show()


def experiment_tree_sin(show_model=True, show_rmse=True):
    data_sin_train, _ = generate_sin_data(n=20, scale=0.2)
    data_sin_test, sin_test_rmse = generate_sin_data(n=1000, scale=0.2)
    rng = np.random.RandomState(1)
    generate_plot(data_sin_train, data_sin_test, tattr='t',
                  model_cls=RegressionTree,
                  iterate_over='max_depth', iterate_values=range(15),
                  title='Regression Tree (sin)',
                  xlabel='max depth',
                  bayes_rmse=sin_test_rmse,
                  rng=rng,
                  show_model=show_model,
                  show_rmse=show_rmse,
                  )
    # plt.savefig('regression_tree_sin.pdf')
    # if show: plt.show()


def experiment_forest_sin(show_model=True, show_rmse=True, multithread = False):
    data_sin_train, _ = generate_sin_data(n=20, scale=0.2)
    data_sin_test, sin_test_rmse = generate_sin_data(n=1000, scale=0.2)
    rng = np.random.RandomState(1)

    generate_plot(data_sin_train, data_sin_test, tattr='t',
                  model_cls=RandomForest,
                  iterate_over='n_trees',
                  iterate_values=[1, 2, 5, 10, 50, 100, 200, 500, 1000],
                  # iterate_values=[10],
                  title='Random Forest (sin)',
                  xlabel='trees count',
                  bayes_rmse=sin_test_rmse,
                  rng=rng,
                  show_model=show_model,
                  show_rmse=show_rmse,
                  multithread = multithread
                  )


def experiment_tree_housing(show_model=True, show_rmse=True):
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=RegressionTree,
                  iterate_over='max_depth', iterate_values=range(15),
                  title='Regression Tree (housing)',
                  xlabel='max depth',
                  rng=rng,
                  show_model=show_model,
                  show_rmse=show_rmse,
                  )


def experiment_forest_housing(show_model=True, show_rmse=True, multithread = False):
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=RandomForest,
                  iterate_over='n_trees',
                  iterate_values=[1, 2, 5, 10, 50, 100, 200, 500, 1000],
                  #iterate_values=[5],
                  title='Random Forest (housing)',
                  xlabel='trees count',
                  rng=rng,
                  show_model=show_model,
                  show_rmse=show_rmse,
                  multithread=multithread
                  )


def experiment_forest_housing_p(show_model=True, show_rmse=True, multithread = False):
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=RandomForest,
                  iterate_over='n_trees',
                  #iterate_values=[1, 2, 5, 10, 50, 100, 200, 500, 1000],
                  iterate_values=[100],
                  title='Random Forest (housing)',
                  xlabel='trees count',
                  rng=rng,
                  show_model=show_model,
                  show_rmse=show_rmse,
                  multithread=multithread
                  )


def experiment_forest_housing_f(show_rmse=True):
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=RandomForest,
                  iterate_over='max_features',
                  iterate_values=[lambda n: 2, lambda n: n / 2, lambda n: n],
                  iterate_labels=['2', '0.5*n', 'n'],
                  title='Random Forest (attr) (housing)',
                  xlabel='attribute count',
                  rng=rng,
                  show_model=False,
                  show_rmse=show_rmse,
                  )


def experiment_gbt_sin(show_model=True, show_rmse=True):
    data_sin_train, _ = generate_sin_data(n=20, scale=0.2)
    data_sin_test, sin_test_rmse = generate_sin_data(n=1000, scale=0.2)
    rng = np.random.RandomState(1)

    generate_plot(data_sin_train, data_sin_test, tattr='t',
                  model_cls=GradientBoostedTrees,
                  iterate_over='n_trees',
                  iterate_values=[1, 2, 5, 10, 50, 100, 200, 500, 1000],
                  title='GBT(sin)[b=1]',
                  xlabel='trees count',
                  bayes_rmse=sin_test_rmse,
                  rng=rng,
                  show_model=show_model,
                  show_rmse=show_rmse,
                  beta=1
                  )


def experiment_gbt_housing(show_rmse=True):
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=GradientBoostedTrees,
                  iterate_over='n_trees',
                  iterate_values=[1, 2, 5, 10, 50, 100, 200, 500, 1000],
                  title='GBT(housing)[b=1]',
                  xlabel='trees count',
                  rng=rng,
                  show_model=False,
                  show_rmse=show_rmse,
                  beta=1
                  )


if __name__ == '__main__':
    # experiment_tree_sin(show_model=True, show_rmse=True)
    #experiment_forest_sin(show_model=True, show_rmse=True, multithread = False)
    experiment_forest_housing_p(show_model=True, show_rmse=True, multithread = True)
    # experiment_tree_housing(show_model=False, show_rmse=True)
    # experiment_forest_housing(show_model=False, show_rmse=True)
    # experiment_forest_housing_f(show_rmse=True)
    # experiment_gbt_sin(show_model=True, show_rmse=True)
    # experiment_gbt_housing(show_rmse=True)
