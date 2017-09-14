import Membership
import FuzzySet
import Rule
import numpy as np

class FuzzySystem(object):

    def __init__(self, name, fuzzysets):
        self.name = name
        self.fuzzysets = fuzzysets
        self.ruleDictionary = {}

    def train(self, X_train, y_train):

        self.createRules(X_train, y_train)

    def createRules(self, X_train, y_train):

        for i in range(X_train.shape[0]):
            pattern_memberships = list()
            pattern_indexes = list()

            for j in range(X_train.shape[1]):
                feat_val = X_train[i, j]

                mu_val, ind_val = FuzzySet.getMaxMembership(feat_val, self.fuzzysets[j])
                pattern_memberships.append(mu_val)
                pattern_indexes.append(ind_val)

            # multiplicar as pertinencias de cada atributo
            rule_membership = 1

            for m in pattern_memberships:
                rule_membership *= m

            # transformar vetor de indices em string (ela será a rule key)


            # (se nao existir) adicionar uma classe rule somando a pertinencia no vetor que representa a class em questao (y_train[i])
            rule_key = str(pattern_indexes)

            if rule_key in self.ruleDictionary:
                self.ruleDictionary[rule_key].updateMembership(y_train[i], rule_membership)
            else:
                rule = Rule.Rule(pattern_indexes, y_train[i], rule_membership)
                self.ruleDictionary[rule_key] = rule

    def test(self, X_test, y_test):

        consequents = list()

        for i in range(X_test.shape[0]):
            consequents.append(self.classifyWithCertaintyGrade(X_test[i,:]))

        errorRate = self.getErrorRate(consequents, y_test)

        return consequents, errorRate

    def classifyWithCertaintyGrade(self, X_test):

        ruleResults = {}

        for rule in self.ruleDictionary.values():

            att_mu = []
            for j in range(len(X_test)):
                ai = rule.antecedent[j]
                fs = self.fuzzysets[j][ai]
                att_mu.append(fs.membership(X_test[j]))

            rule_membership = 1
            for m in att_mu:
                rule_membership *= m

            ruleResults[rule.getKey()] = rule_membership * rule.getCertaintyGrade()

        max_rule_key = max(ruleResults.keys(), key=(lambda k: ruleResults[k]))

        return self.ruleDictionary[max_rule_key].consequent



    def getErrorRate(self, returned, expected):

        error = sum(i != j for i, j in zip(returned, expected))

        return (error / len(expected)) * 100