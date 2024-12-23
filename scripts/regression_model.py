from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump

def train_regression_model(X, y):
    """
    Train a regression model to predict satisfaction scores.

    Args:
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable (satisfaction scores).

    Returns:
        tuple: Trained model, training loss.
    """
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    loss = mean_squared_error(y, predictions)
    dump(model, "../models/regression_model.pkl")
    return model, loss
