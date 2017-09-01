import Membership
import FuzzySet
import numpy as np

class Partitioner(object):
    """
    Universe of Discourse partitioner. Split data on several fuzzy sets
    """

    def __init__(self, name, data, npart, func=Membership.trimf, names=None, prefix="A", transformation=None, indexer=None):
        """
        Universe of Discourse partitioner scheme. Split data on several fuzzy sets
        :param name: partitioner name
        :param data: original data to be partitioned
        :param npart: number of partitions
        :param func: membership function
        :param names: list of partitions names. If None is given the partitions will be auto named with prefix
        :param prefix: prefix of auto generated partition names
        :param transformation: data transformation to be applied on data
        """
        self.name = name
        self.partitions = npart
        self.sets = []
        self.membership_function = func
        self.setnames = names
        self.prefix = prefix
        self.transformation = transformation
        self.indexer = indexer

        if self.indexer is not None:
            ndata = self.indexer.get_data(data)
        else:
            ndata = data

        if transformation is not None:
            ndata = transformation.apply(ndata)
        else:
            ndata = data

        _min = min(ndata)
        if _min < 0:
            self.min = _min * 1.1
        else:
            self.min = _min * 0.9

        _max = max(ndata)
        if _max > 0:
            self.max = _max * 1.1
        else:
            self.max = _max * 0.9

        self.sets = self.build(ndata)

        del(ndata)

    def build(self, data):
        sets = []

        dlen = self.max - self.min
        partlen = dlen / self.partitions

        count = 0
        for c in np.arange(self.min, self.max, partlen):
            sets.append(
            FuzzySet.FuzzySet(self.prefix + str(count), Membership.trimf, [c - partlen, c, c + partlen],c))
            count += 1

        return sets

    def plot(self, ax):
        """
        Plot the
        :param ax:
        :return:
        """
        ax.set_title(self.name)
        ax.set_ylim([0, 1])
        ax.set_xlim([self.min, self.max])
        for s in self.sets:
            if s.mf == Membership.trimf:
                ax.plot([s.parameters[0], s.parameters[1], s.parameters[2]], [0, 1, 0])
            elif s.mf == Membership.gaussmf:
                tmpx = [kk for kk in np.arange(s.lower, s.upper)]
                tmpy = [s.membership(kk) for kk in np.arange(s.lower, s.upper)]
                ax.plot(tmpx, tmpy)

    def __str__(self):
        return self.name
