import pandas as pd
import numpy as np
from joblib import load


def predict_and_average(X, models_paths):
    predictions = 0
    preds = []
    for model_filename in models_paths:
        model = load(model_filename)
        features = model.feature_names_
        pred = model.predict(X[features])
        preds.append(pred)
        predictions += pred
    
    predictions /= len(models_paths)
    return predictions


def fill_na_from_lookup(test_df, features, avgs_for_age_group):
    """
    Fills NaN values in specified features of test_df with averages from avgs_for_age_group,
    matched by 'Age Group'.

    Parameters:
    - test_df: pandas.DataFrame, the DataFrame with data to be imputed.
    - features: list of str, the features in test_df for which NaN values should be filled.
    - avgs_for_age_group: pandas.DataFrame, the lookup table with averages for each age group,
                          'Age Group' should be the index of this DataFrame.
    """
    test_df_filled = test_df.copy()
    
    for feature in features:
        # Iterate over each unique age group in the test dataframe
        for age_group in test_df_filled['Age Group'].unique():
            # Check if the age group is in the avgs_for_age_group index
            if age_group in avgs_for_age_group.index:
                # Fill NaN values for this feature and age group
                # with the corresponding average from the lookup table
                mask = (test_df_filled['Age Group'] == age_group) & (test_df_filled[feature].isna())
                test_df_filled.loc[mask, feature] = avgs_for_age_group.loc[age_group, feature]
    return test_df_filled


def remove_young_data(df, age_column='age', age_threshold=20, removal_fraction=0.2):
    """
    Randomly removes a specified fraction of data for rows where the age is below a certain threshold.

    Parameters:
    - df: pandas.DataFrame, the original dataset.
    - age_column: str, the name of the column in the DataFrame that contains age data.
    - age_threshold: int, the age below which rows are considered for random removal.
    - removal_fraction: float, the fraction of the targeted rows to remove.

    Returns:
    - df_reduced: pandas.DataFrame, the dataset after random removal of the specified fraction of targeted rows.
    """
    # Identify rows where age is below the specified threshold
    young_indices = df[df[age_column] < age_threshold].index
    
    # Calculate the number of rows to remove
    num_to_remove = int(np.floor(len(young_indices) * removal_fraction))
    
    # Randomly select a subset of these rows to remove
    remove_indices = np.random.choice(young_indices, size=num_to_remove, replace=False)
    
    # Drop the selected rows
    df_reduced = df.drop(remove_indices)
    
    return df_reduced

def upscale_with_nans(df, features, missing_fraction=0.4, upscale_factor=1):
    """
    Augments and upscales the input DataFrame by appending new rows with artificial NaNs,
    ensuring each feature is missing at least `missing_fraction` of the times in the augmented data.
    The size of the augmented portion is controlled by `upscale_factor`.

    Parameters:
    - df: pandas.DataFrame, the original dataset.
    - missing_fraction: float, the fraction of missing values to introduce per feature in the augmented data.
    - upscale_factor: int, the factor by which to upscale the dataset (e.g., 1 means doubling the size).

    Returns:
    - df_upscaled: pandas.DataFrame, the original dataset with the augmented data appended.
    """
    augmented_dfs = []

    for _ in range(upscale_factor):
        augmented_df = df.copy()
        
        # Calculate the number of missing values to introduce per feature
        num_rows, num_features = df.shape
        missing_per_feature = int(np.ceil(num_rows * missing_fraction))
        
        for feature in features:
            # Randomly select rows to introduce NaNs
            nan_indices = np.random.choice(df.index, size=missing_per_feature, replace=False)
            
            # Introduce NaNs
            augmented_df.loc[nan_indices, feature] = np.nan
        
        augmented_dfs.append(augmented_df)

    # Concatenate the original df with all augmented copies
    df_upscaled = pd.concat([df] + augmented_dfs, ignore_index=True)
    
    return df_upscaled

