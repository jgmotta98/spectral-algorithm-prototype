import pandas as pd
from sklearn.linear_model import SGDClassifier


def svg_algorithm(df_tuple: tuple[pd.DataFrame, pd.DataFrame], *, verbose: bool = False) -> None:
    test_df, train_df = df_tuple

    train_df_x = train_df.drop(columns=['Name'])
    train_df_y = train_df['Name']
    
    test_df_x = test_df.drop(columns=['Name'])

    svg = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    svg.fit(train_df_x, train_df_y)

    probabilities = svg.predict_proba(test_df_x)
    classes = svg.classes_
    output = {cls: proba for cls, proba in zip(classes, probabilities[0])}
    sorted_output = sorted(output.items(), key=lambda x: x[1], reverse=True)

    similar_compounds = [output[0] for output in sorted_output[:5]]

    if verbose:
        print(f'SVG Results: {similar_compounds}')