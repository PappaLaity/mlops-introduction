from functions.iris_functions import evaluate_iris_lr_model, load_iris_data, train_iris_lr_model
from sklearn.linear_model import LogisticRegression



def test_load_iris_data():
    X_train, X_test, y_train, y_test = load_iris_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

def test_train_iris_lr_model():
    X_train, _, y_train, _ = load_iris_data()
    lr_model = train_iris_lr_model(X_train,y_train)

    assert isinstance(lr_model, LogisticRegression)

    

def test_evaluate_iris_lr_model():
    # Check that model can predict
    # y_pred = model.predict(X_test)
    X_train, X_test, y_train, y_test = load_iris_data()
    lr_model = train_iris_lr_model(X_train,y_train)
    y_pred = lr_model.predict(X_test)
    accuracy_score = evaluate_iris_lr_model(y_pred,y_test)
    assert len(y_pred) == len(y_test)
    assert accuracy_score >= 0.7