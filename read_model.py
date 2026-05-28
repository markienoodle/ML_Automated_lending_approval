import joblib
loaded = joblib.load("cart_outputs/cart_final_model.pkl")
print(loaded["best_params"])
print(loaded["test_auc"])