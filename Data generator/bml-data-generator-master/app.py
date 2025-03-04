from typing import Literal

import pandas as pd

from generators import *

# Initialize a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(11)


def create_dataframe(n_rows: int, n_columns: int) -> pd.DataFrame:
    """
    Creates a pandas DataFrame filled with random floats.

    Parameters:
    - n_rows (int): Number of rows in the DataFrame.
    - n_columns (int): Number of columns in the DataFrame.

    Returns:
    - pd.DataFrame: DataFrame filled with random floats between -0.1 and 0.1.
    """
    return pd.DataFrame(rng.uniform(-1, 1, (n_rows, n_columns)))


def set_categorical_features(df: pd.DataFrame, categorical_ratio: float) -> pd.DataFrame:
    """
    Modifies a given DataFrame to include random categorical features.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - categorical_ratio (float): The ratio of columns to convert to categorical.

    Returns:
    - pd.DataFrame: The modified DataFrame with categorical features.
    """
    categorical_idx = rng.choice(df.shape[1], size=int(df.shape[1] * categorical_ratio), replace=False)
    for i in categorical_idx:
        df[i] = np.random.choice([0, 1], size=df.shape[0], p=[19 / 20, 1 / 20]).astype(np.uint8)
    return df


def get_informative_indices(df: pd.DataFrame, informative_ratio: float) -> np.ndarray:
    """
    Selects a subset of columns to be considered informative.

    Parameters:
    - df (pd.DataFrame): DataFrame from which to select informative columns.
    - informative_ratio (float): The ratio of columns to be considered informative.

    Returns:
    - np.ndarray: Array of indices for informative columns.
    """
    return rng.choice(df.shape[1], size=int(df.shape[1] * informative_ratio), replace=False)


def get_intercept(low: float, high: float) -> float:
    """
    Generates a random intercept value within a specified range.

    Parameters:
    - low (float): The lower bound of the intercept range.
    - high (float): The upper bound of the intercept range.

    Returns:
    - float: A random intercept value.
    """
    return np.random.uniform(low, high)


def compute_y_exact(df: pd.DataFrame, intercept: float, coefficients: np.ndarray) -> pd.DataFrame:
    """
    Computes the exact y values for a regression dataset.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the features.
    - intercept (float): Intercept of the regression equation.
    - informative_indices (np.ndarray): Indices of informative columns.
    - coefficients (np.ndarray): Coefficients for each informative feature.

    Returns:
    - pd.DataFrame: DataFrame with the computed y_exact column.
    """
    df['y_exact'] = intercept
    for idx, c in enumerate(coefficients):
        df['y_exact'] += c * df.iloc[:, idx]
    return df


class Generator:
    def __init__(self, n_rows: int, n_columns: int, informative_ratio: float,
                 distribution: Literal['norm', 'poisson', 'gamma'], gaussian_sigma2: np.float32 = 1.):
        self.coef_ = None
        self.informative_indices_ = None
        self.intercept_ = None
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.informative_ratio = informative_ratio
        self.distribution = distribution
        self.gaussian_sigma2 = gaussian_sigma2

    def generate_data(self):
        # Step 1: Create a DataFrame with random floats
        df = create_dataframe(self.n_rows, self.n_columns)

        # Step 2: Set random categorical features
        # df = set_categorical_features(df, self.categorical_ratio)

        # Step 3: Set random intercept
        self.intercept_ = get_intercept(-1, 1)

        # Step 4: Identify informative indices
        self.informative_indices_ = get_informative_indices(df, self.informative_ratio)

        # Step 5: Generate random standardized coefficients
        self.coef_ = np.random.uniform(-1, 1, size=self.n_columns)
        for idx in range(len(self.coef_)):
            if idx not in self.informative_indices_:
                self.coef_[idx] = 0

        # Step 6: Compute exact y values
        df = compute_y_exact(df, self.intercept_, self.coef_)

        if self.distribution == 'norm':
            df['y'] = generate_gaussian(df['y_exact'], self.gaussian_sigma2)
        elif self.distribution == 'poisson':
            df['y'] = generate_poisson(df['y_exact'])
        elif self.distribution == 'gamma':
            df['y'] = generate_gamma(df['y_exact'])

        return df
