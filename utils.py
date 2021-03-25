import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def generate_poly_features(df, cols, max_degree=3):
    """
    Function to add polynomial transformation (x -> x^i) for column names in cols (iterable) in df (Dataframe)
    max_degree - maximum degree of polynomial features to be added
    """
    df = df.copy()
    for col in cols:
        for degree in range(1, max_degree+1):
            df[f"{col}^{degree}"] = df[col]**degree
    return df 

def summary_model(estimator, summary_name, X, y):
    
    y_pred = estimator.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"The model performance for {summary_name}")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

    



