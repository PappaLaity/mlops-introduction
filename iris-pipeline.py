from functions.iris_functions import evaluate_iris_lr_model, load_iris_data, train_iris_lr_model


X_train, X_test, y_train, y_test = load_iris_data()

lr_model = train_iris_lr_model(X_train,y_train)

y_pred = lr_model.predict(X_test)

accuracy = evaluate_iris_lr_model(y_test,y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

