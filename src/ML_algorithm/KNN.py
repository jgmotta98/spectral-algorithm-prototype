from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def KNN_algorithm(df_tuple: tuple[pd.DataFrame, pd.DataFrame], *, verbose: bool = False) -> None:
    similar_compounds: list[str] = []
    test_df, train_df = df_tuple

    train_df_x = train_df.drop(columns=['Name'])
    train_df_y = train_df['Name']
    
    test_df_x = test_df.drop(columns=['Name'])
    
    for _ in range(5):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_df_x, train_df_y)

        y_pred = knn.predict(test_df_x)
        similar_compounds.append(y_pred[0])

        similar_index = train_df_y[train_df_y == y_pred[0]].index[0]
        train_df_y = train_df_y.drop(similar_index)
        train_df_x = train_df_x.drop(similar_index)

    if verbose:
        print(f'KNN Results: {similar_compounds}')