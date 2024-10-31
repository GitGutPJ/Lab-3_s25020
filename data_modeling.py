import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def fetch_data(url):
    return pd.read_csv(url)


def prepare_data(df):
    # Standaryzacja
    scaler = StandardScaler()

    num_columns = df.select_dtypes(include=['int64', 'float64']).columns

    df[num_columns] = scaler.fit_transform(df[num_columns])

    model_training(df)


def model_training(df):
    X = df.drop(['rownames', 'score'], axis=1)
    Y = df['score']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=2024)

    model = RandomForestRegressor(random_state=2024)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    with open(f'RandomForestRegressor.txt', 'w') as f:
            f.write(f'R2: {r2}\nMSE: {mse}\nMAE: {mae}')

    param_grid_rf = {
        'n_estimators': [50, 100, 150, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [5, 10, 20, None],
    }

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid=param_grid_rf, cv=5)

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    with open(f'RandomForestRegressor_optimize_params.txt', 'w') as f:
        f.write(f'{grid_search.best_estimator_}')

    y_best_rf = best_rf.predict(X_test)

    best_r2_rf = r2_score(y_test, y_best_rf)
    best_mse_rf = mean_squared_error(y_test, y_best_rf)
    best_mae_rf = mean_absolute_error(y_test, y_best_rf)

    with open(f'RandomForestRegressor_optimize.txt', 'w') as f:
        f.write(f'R2: {best_r2_rf}\nMSE: {best_mse_rf}\nMAE: {best_mae_rf}')


if __name__ == '__main__':
    df = fetch_data('filtredCollegeDistance.csv')
    prepare_data(df)
