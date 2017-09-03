import numpy as np

class Rule:
    def __init__(self, antecedent, consequent, membership):

        self.antecedent = antecedent
        self.membership = membership
        self.consequent = consequent
        self.certainty_grade = 0
        self.beta = np.zeros(3)
        self.beta[consequent] = membership

    def getKey(self):
        return str(self.antecedent)

    def updateMembership(self, consequent, membership):
        self.beta[consequent] += membership

    def getCertaintyGrade(self):

        if self.certainty_grade == 0:
            max_beta = max(self.beta)
            max_beta_index = np.argmax(self.beta)

            other_betas = 0

            for i in range(self.beta.shape[0]):
                if i != max_beta_index:
                    other_betas += self.beta[i]

            self.certainty_grade = (max_beta - other_betas) / (max_beta + other_betas)

        return self.certainty_grade


