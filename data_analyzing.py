import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def read_CSV(url):
    df = pd.read_csv(url)
    return df


def clean_and_analyze(df):
    print(f'Ilość wierszy i kolumn: {df.shape}')
    print(f'Ilość brakujących danych: {df.isna().sum()}')
    print(f'Statystyka zmiennych num:\n {df.describe}')

    df.dropna(thresh=5)

    numeric_imputer = SimpleImputer(strategy='mean')
    categoric_imputer = SimpleImputer(strategy='most_frequent')

    df[['rownames', 'score', 'unemp', 'wage', 'distance', 'tuition', 'education']] = numeric_imputer.fit_transform(
        df[['rownames', 'score', 'unemp', 'wage', 'distance', 'tuition', 'education']])
    df[['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income',
        'region']] = categoric_imputer.fit_transform(
        df[['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']])

    num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

    graphs_gen_num(num_columns)
    graphs_gen_cat(cat_columns)

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[cat_columns])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(cat_columns))

    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(cat_columns,axis=1)

    #heatmap(df_encoded)

    return df

def graphs_gen_num(num_columns):
    for col in num_columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.xlabel(col)
        plt.ylabel('Liczność')
        plt.savefig(f'graphs/{col}.png')
        plt.close()


def graphs_gen_cat(cat_columns):
    for col in cat_columns:
        plt.figure()
        sns.countplot(df[col])
        plt.xlabel('Liczność')
        plt.ylabel(col)
        plt.savefig(f'graphs/{col}.png')
        plt.close()


def heatmap(df_encoded):
    plt.figure(figsize=(15, 15))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.2)
    plt.savefig(f'graphs/heatmap.png')
    plt.close()


if __name__ == '__main__':
    df = read_CSV('https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv')
    df_encoded = clean_and_analyze(df)
    df_encoded.to_csv('filtredCollegeDistance.csv')