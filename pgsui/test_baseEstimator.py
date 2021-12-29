import numpy as np
import sklearn as sk

from sklearn.linear_model import LinearRegression, LassoLarsCV, RidgeCV
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator

def main():
    ### Fit the c parameter ###
    X = np.random.normal(0, 1, (100,5))
    y = X[:,1] * X[:,2] + np.random.normal(0, .1, 100)

    gs = sk.grid_search.GridSearchCV(ELM(n_nodes=20, output_function='lr'),
                                     cv=5,
                                     param_grid={"c":np.linspace(0.0001,1,10)},
                                     fit_params={})



class ELM(BaseEstimator):

    def __init__(self, n_nodes, link='rbf', output_function='lasso', n_jobs=1, c=1):
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        self.c = c

        if link == 'rbf':
            self.link = lambda z: np.exp(-z*z)
        elif link == 'sig':
            self.link = lambda z: 1./(1 + np.exp(-z))
        elif link == 'id':
            self.link = lambda z: z
        else:
            self.link = link

        if output_function == 'lasso':
            self.output_function = LassoLarsCV(cv=10, n_jobs=self.n_jobs)
        elif output_function == 'lr':
            self.output_function = LinearRegression(n_jobs=self.n_jobs)

        elif output_function == 'ridge':
            self.output_function = RidgeCV(cv=10)

        else:
            self.output_function = output_function

        return


    def H(self, x):

        n, p = x.shape
        xw = np.dot(x, self.w.T)
        xw = xw + np.ones((n, 1)).dot(self.b.T)
        return self.link(xw)

    def fit(self, x, y, w=None):

        n, p = x.shape
        self.mean_y = y.mean()
        if w == None:
            self.w = np.random.uniform(-self.c, self.c, (self.n_nodes, p))
        else:
            self.w = w

        self.b = np.random.uniform(-self.c, self.c, (self.n_nodes, 1))
        self.h_train = self.H(x)
        self.output_function.fit(self.h_train, y)

        return self

    def predict(self, x):
        self.h_predict = self.H(x)
        return self.output_function.predict(self.h_predict)

    def get_params(self, deep=True):
        return {"n_nodes": self.n_nodes,
                "link": self.link,
                "output_function": self.output_function,
                "n_jobs": self.n_jobs,
                "c": self.c}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)



if __name__ == "__main__":
    main()
