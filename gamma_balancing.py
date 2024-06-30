"""
Author: Rohan Mitra (rohanmitra8@gmail.com)
gamma_balancing.py (c) 2024
Desc: description
Created:  2024-06-26T11:43:13.956Z
Modified: 2024-06-30T21:42:31.745Z
"""

import pandas as pd
import numpy as np
import os
from time import time
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class GammaBalancerBinary:

    def __init__(self, df, target_col, majority_class=1, minority_class=0, categorical_col_names=[]):
        self.df = df
        self.target_col = target_col
        self.shape = 0  # Just defaulting values
        self.scale = 0  # Will be calculated when find_best_params is called
        self.categorical_col_names = categorical_col_names
        self.num_col_names = [x for x in df.columns if x not in categorical_col_names and x != target_col]
        
        #Check if majority class is indeed the majority class - incase of typos
        if df[target_col].value_counts()[majority_class]<df[target_col].value_counts()[minority_class]:
            #Majority class and minority class are swapped, so swap them back:
            majority_class,minority_class = minority_class, majority_class
        
        self.majority_class = majority_class
        self.minority_class = minority_class

    # Function to calculate the required values and generate the random point
    def calculate_values_and_generate_point_numerical(self, x1, x2, shape, scale):
        direction = x2 - x1
        norm = np.linalg.norm(direction)
        if norm == 0:
            return x1  # If the direction is zero, return the original point
        u = direction / norm
        t = np.random.gamma(shape, scale)
        mode = (shape - 1) * scale
        z = x1 + (t - mode) * u
        return z

    def calculate_values_and_generate_point_categorical(self, p1, p2, shape, scale):
        # For categorical columns, just pick the same value as p1
        cat_df = p1[self.categorical_col_names]

        # For numerical columns, do the gamma method
        # x1 and x2 are dfs not np arrays
        x1 = p1[self.num_col_names]
        x2 = p2[self.num_col_names]
        direction = x2 - x1
        norm = np.linalg.norm(direction)
        if norm == 0:
            return p1  # If the direction is zero, return the original point
        u = direction / norm
        t = np.random.gamma(shape, scale)
        mode = (shape - 1) * scale
        z = x1 + (t - mode) * u

        # z here is the new point for numerical columns as a df
        ans = pd.concat(
            [
                pd.DataFrame([cat_df.values], columns=self.categorical_col_names),
                pd.DataFrame([z.values], columns=self.num_col_names),
            ],
            axis=1,
        )
        return ans[self.df.columns[:-1]].values[0]

    def generate_point(self, args):
        x1, x2, shape, scale = args
        if len(self.categorical_col_names) == 0:
            return self.calculate_values_and_generate_point_numerical(x1, x2, shape, scale)
        else:
            return self.calculate_values_and_generate_point_categorical(x1, x2, shape, scale)


    def oversample_minority_class_cat(self, df=None, shape=None, scale=None):

        # Allows the user to use the function without passing in any arguments since all these are already stored in the object
        if df is None:
            df = self.df
        if shape is None:
            shape = self.shape
        if scale is None:
            scale = self.scale

        minority_class = df[df[self.target_col] == self.minority_class]
        majority_class = df[df[self.target_col] == self.majority_class]
        num_minority = len(minority_class)
        num_majority = len(majority_class)
        num_samples_needed = num_majority - num_minority
        new_samples = []
        
        minority_class_values = minority_class.drop(self.target_col, axis=1)

        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(minority_class_values)

        idx1 = np.random.choice(num_minority, num_samples_needed)
        # pick all points for x1 and x2
        x1 = minority_class_values.iloc[idx1]

        distances, indices = nn.kneighbors(x1.values, n_neighbors=2)
        idx2 = indices[:, 1]  # All rows, 2nd nearest neighbor
        x2 = minority_class_values.iloc[idx2]

        # create a list of arguments
        args = [(x1.iloc[i], x2.iloc[i], shape, scale) for i in range(num_samples_needed)]

        new_samples = [self.generate_point(arg) for arg in args]

        new_samples_df = pd.DataFrame(new_samples, columns=minority_class.columns[:-1])
        new_samples_df[self.target_col] = self.minority_class
        df_oversampled = pd.concat([self.df, new_samples_df], ignore_index=True)
        #!TBD: Should i put this into self.df?
        return df_oversampled

    def oversample_minority_class_num(self, df=None, shape=None, scale=None):
        # Allows the user to use the function without passing in any arguments since all these are already stored in the object
        if df is None:
            df = self.df
        if shape is None:
            shape = self.shape
        if scale is None:
            scale = self.scale

        minority_class = df[df[self.target_col] == self.minority_class]
        majority_class = df[df[self.target_col] == self.majority_class]
        num_minority = len(minority_class)
        num_majority = len(majority_class)
        num_samples_needed = num_majority - num_minority
        new_samples = []
        
        minority_class_values = minority_class.drop(self.target_col, axis=1).values

        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(minority_class_values)

        idx1 = np.random.choice(num_minority, num_samples_needed)
        # pick all points for x1 and x2
        x1 = minority_class_values[idx1]

        distances, indices = nn.kneighbors(x1, n_neighbors=2)
        idx2 = indices[:, 1]  # All rows, 2nd nearest neighbor
        x2 = minority_class_values[idx2]

        # create a list of arguments
        args = [(x1[i], x2[i], shape, scale) for i in range(num_samples_needed)]

        new_samples = [self.generate_point(arg) for arg in args]

        new_samples_df = pd.DataFrame(new_samples, columns=minority_class.columns[:-1])
        new_samples_df[self.target_col] = self.minority_class
        df_oversampled = pd.concat([self.df, new_samples_df], ignore_index=True)
        #!TBD: Should i put this into self.df?
        return df_oversampled

    # Function to oversample the minority class
    def oversample_minority_class(self, df=None, shape=None, scale=None):
        if len(self.categorical_col_names) == 0:
            return self.oversample_minority_class_num(df, shape, scale)
        else:
            return self.oversample_minority_class_cat(df, shape, scale)
        

    ###############################################################################################################################
    # Functions to find the best shape and scale parameters using Grid Search

    def evaluate_oversampling(self, shape, scale, df_train, X_test, y_test, minority_class, majority_class):
        print("Oversampling minority class")
        t0 = time()
        df_train_oversampled = self.oversample_minority_class(df_train, shape, scale)
        print("Time taken for oversampling:", time() - t0)
        X_train_oversampled = df_train_oversampled.drop(self.target_col, axis=1)
        y_train_oversampled = df_train_oversampled[self.target_col]

        # Use random forest classifier:
        model = RandomForestClassifier()
        model.fit(X_train_oversampled, y_train_oversampled)

        # Predict and evaluate on the test set for balanced training data
        y_pred_balanced = model.predict(X_test)
        report_balanced = classification_report(y_test, y_pred_balanced, output_dict=True)
        macro_avg_f1_score = report_balanced["macro avg"]["f1-score"]
        return macro_avg_f1_score

    def find_best_params(self, X_train, X_test, y_train, y_test, parameter_sets=[], verbose=True):
        # Shape must be >= scale param
        # Parameter sets = [(shape, scale)]
        if any(x[0] < x[1] for x in parameter_sets):
            # If any scale is greater than shape
            raise ValueError("Shape must be >= scale")

        # Raise error incase parameter_sets are empty
        if len(parameter_sets) == 0:
            raise ValueError("Parameter sets cannot be empty")

        # Combine X_train and y_train into a single DataFrame
        df_train = pd.concat([X_train, y_train], axis=1)

        # Evaluate each parameter set
        results = {}
        t0 = time()
        for shape, scale in parameter_sets:
            if verbose:
                print(f"Doing {shape=},{scale=}")
            t1 = time()
            f1_score = self.evaluate_oversampling(
                shape, scale, df_train, X_test, y_test, self.minority_class, self.majority_class
            )
            if verbose:
                print("Time taken:", time() - t1)
            results[(shape, scale)] = f1_score
        if verbose:
            print("Time taken to evaluate all oversamplings:", time() - t0)
            print()
            print(results)
            print()
        sorted_results = list(sorted(results.items(), key=lambda x: x[1], reverse=True))
        best_params = sorted_results[0]
        shape, scale = best_params[0]
        if verbose:
            print(f"Best shape:{shape}, best scale:{scale}")
        self.shape = shape
        self.scale = scale
        return shape, scale


if __name__ == "__main__":
    ############ SAMPLE USAGE
    parameter_sets = [(2, 2), (3, 2), (5, 1), (9, 0.5), (7, 1), (2, 1), (3, 1), (4, 1), (3, 3)]

    df = pd.read_csv("./Kaggle Dataset/Kaggle_data_preprocessed_normalized.csv") #Replace with your dataset
    # Assumes df is already processed - contains only X and y, and is already label encoded and scaled appropriately
    target_col = "class"
    categorical_col_names = ["protocol_type", "service", "flag"]
    print(df[target_col].value_counts())

    balancer = GammaBalancerBinary(
        df, target_col, majority_class=1, minority_class=0, categorical_col_names=categorical_col_names
    )

    # Split data into train and test split to find best params
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split into train val and test set
    X_train_Val, X_test, y_train_Val, y_test = train_test_split(X, y, test_size=0.2)

    # Split into train and val sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_Val, y_train_Val, test_size=0.2)

    # Use the validation set to find the best shape and scale params
    best_shape, best_scale = balancer.find_best_params(X_train, X_val, y_train, y_val, parameter_sets=parameter_sets)

    # Now we oversample the minority class based on the shape and scale params found

    # Combine x_train and y_train into a single df
    df_train = pd.concat([X_train, y_train], axis=1)

    t0 = time()
    df_train_oversampled = balancer.oversample_minority_class(df_train, best_shape, best_scale)
    print("Time taken for oversampling:", time() - t0)

    # Split the new oversampled train dataset back into xtrain and ytrain
    X_train_oversampled = df_train_oversampled.drop(target_col, axis=1)
    y_train_oversampled = df_train_oversampled[target_col]

    # Use random forest classifier:
    model = RandomForestClassifier()
    model.fit(X_train_oversampled, y_train_oversampled)

    # Predict and evaluate on the test set
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_avg_f1_score = report["macro avg"]["f1-score"]
    print("Macro Average F1 Score:", macro_avg_f1_score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
