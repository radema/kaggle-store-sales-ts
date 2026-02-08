from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


class ScaledTrendRegressor(Pipeline):
    """
    Standard Linear Regression with Scaling.
    Using LinearRegression instead of Ridge to ensure the model captures 100%
    of the volume signal without regularization bias.
    """

    def __init__(self):
        steps = [("linear", LinearRegression())]
        super().__init__(steps)

    def get_params(self, deep=True):
        return super().get_params(deep=deep)
