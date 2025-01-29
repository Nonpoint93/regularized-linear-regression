import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_regression
import joblib


# your code here
alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

def read_csv() -> pd.DataFrame:
   try:
       return pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv',
                           delimiter=',',
                           skipinitialspace=True,
                           engine='python')
   except Exception as e:
       print(e)


def save_best_model(model, alpha, r2_scores):
    joblib.dump(model, f'./models/best_lasso_alpha_{alpha}.pkl')
    print(f"Best model saved with alpha={alpha} and R²={max(r2_scores):.4f}")


def main():
    dataframe: pd.DataFrame = read_csv()
    print(dataframe.columns)
    print(dataframe.shape)
    dataframe.info()
    print(dataframe.describe(exclude=[object]))

    if dataframe.duplicated().sum() > 0:
        print(f"[+] There are {dataframe.duplicated().sum()} duplicated rows")
        dataframe = dataframe.drop_duplicates()
        print(f"[+] Duplicated rows have been removed")
    else:
        print("[+] There are no duplicated rows")

    if dataframe.isna().sum().sum() > 0 or dataframe.isnull().sum().sum() > 0:
        print(f"[+] There are {dataframe.isna().sum().sum()} missing values (NAs)")
        print(f"[+] There are {dataframe.isnull().sum().sum()} missing values (NULLs)")
        dataframe = dataframe.dropna(axis=1)
    else:
        print("[+] There are no missing values")

    dataframe: pd.DataFrame = dataframe.select_dtypes(include=['float64', 'int64'])

    X: pd.DataFrame = dataframe.drop(columns=['diabetes_prevalence'], axis=1)
    y: pd.DataFrame = dataframe['diabetes_prevalence']

    k = 10
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected features: {selected_features}")

    X = selector.transform(X)
    X = pd.DataFrame(X, columns=selected_features)  
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pd.concat([X_train, y_train], axis=1).to_csv('./data/processed/train_data.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('./data/processed/test_data.csv', index=False)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    print(f"Intercept (a): {linear_model.intercept_}")
    print(f"Coefficients (b1, b2): {linear_model.coef_}")

    y_pred = linear_model.predict(X_test)

    print("Linear Regression:")
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
    print(f"Coefficient of determination: {r2_score(y_test, y_pred)}")


    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)

    r2_scores = []
    print("\nLasso Regression:")
    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train, y_train)
        y_pred = lasso_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        print(f"Alpha: {alpha} | MSE: {mean_squared_error(y_test, y_pred):.4f} | R²: {r2:.4f}")
        for alpha_val, r2_val in zip(alphas, r2_scores):
            plt.annotate(f"{r2_val:.2f}", (alpha_val, r2_val), textcoords="offset points", xytext=(0,10), ha='center')

    best_alpha = alphas[r2_scores.index(max(r2_scores))]
    save_best_model(Lasso(alpha=best_alpha).fit(X_train, y_train), best_alpha, r2_scores)

    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.plot(alphas, r2_scores, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Alpha (Lasso) [Log Scale]')    
    plt.show()

if __name__ == '__main__':
    main()