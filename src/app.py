import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso


# your code here
alphas = [0.1, 1.0, 10.0, 20.0]
r2_scores = []

def read_csv() -> pd.DataFrame:
   try:
       return pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv',
                           delimiter=',',
                           skipinitialspace=True,
                           engine='python')
   except Exception as e:
       print(e)



def main():
    dataframe: pd.DataFrame = read_csv()
    print(dataframe.columns)
    print(dataframe.shape)
    print(dataframe.info())
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

    dataframe: pd.DataFrame = dataframe.drop(columns=['fips', 'STATE_FIPS', 'CNTY_FIPS', 'STATE_NAME'], axis=1)
    dataframe: pd.DataFrame = dataframe.select_dtypes(include=['float64', 'int64'])

    X: pd.DataFrame = dataframe.drop(columns=['diabetes_prevalence'], axis=1)
    y: pd.DataFrame = dataframe['diabetes_prevalence']

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

    print("\nLasso Regression:")
    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train, y_train)
        y_pred = lasso_model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        print(f"Alpha: {alpha} | MSE: {mean_squared_error(y_test, y_pred):.4f} | R²: {r2_score(y_test, y_pred):.4f}")


    plt.figure(figsize=(8, 6))
    plt.plot(alphas, r2_scores, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('R² Score')
    plt.title('Evolution of R² with difference values of Alpha in Lasso')
    plt.show()

if __name__ == '__main__':
    main()