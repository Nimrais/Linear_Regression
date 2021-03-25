def generate_poly_features(df, cols, max_degree=3):
    df = df.copy()
    for col in cols:
        for degree in range(max_degree):
            df[f"{col}^{degree}"] = df[col]**degree
    return df 
