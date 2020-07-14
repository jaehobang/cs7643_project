"""
Random forest wrapper

@Jaeho Bang
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from filters.models.ml_base import MLBase


class MLRandomForest(MLBase):
  def __init__(self, **kwargs):
    super().__init__()
    if kwargs:
      self.model = RandomForestClassifier(max_depth=2, random_state=0)
    else:
      self.model = RandomForestClassifier(**kwargs)



  def train(self, X :np.ndarray, y :np.ndarray):
    ## we should control the amount of features, the limit should be around 10000 (100,100)
    X = self._downsize_input(X, number_of_features=10000)

    X = self._flatten_input(X)
    y = self._check_label(y) ## make sure everything is binary labels!!

    n_samples = X.shape[0]
    division = int(n_samples * self.division_rate)
    X_train = X[:division]
    y_train = y[:division]
    X_val = X[division:]
    y_val = y[division:]

    print(X_train.shape)
    print(y_train.shape)
    self.model.fit(X_train, y_train)
    tic = time.time()
    score = self.model.score(X_val, y_val)
    toc = time.time()
    val_samples = X_val.shape[0]
    y_hat = self.model.predict(X_val)

    self.C = (toc - tic) / val_samples
    self.A = score
    self.R = 1 - float(sum(y_hat)) / len(y_hat)


  def predict(self, X :np.ndarray):
    X = self._flatten_input(X)
    return self.model.predict(X)


  def _downsize_input(self, X, number_of_features):
    import math
    wanted_width = int(math.sqrt(number_of_features))
    width_skip_rate = X.shape[1] // wanted_width
    wanted_height = number_of_features // wanted_width
    height_skip_rate = X.shape[2] // wanted_height
    return X[:, ::height_skip_rate, ::width_skip_rate, :]


  def _flatten_input(self, X):
    if X.ndim > 2:
      return X.reshape(X.shape[0], -1)
    elif X.ndim < 2:
      print("Input dimension is less than 2... input shape is wrong")
      return X
    else:
      return X



  def _check_label(self, y:np.ndarray):

    y[y > 1] = 1

    return y
