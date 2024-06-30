# GammaBalancer
Gamma Distribution based synthetic data generator for dataset balancing as proposed in **[Paper Name]**

`Note: Only works for Binary Classification problems for now. Will be updated for Multiclass classification problems soon`

------------------------------------------
# Sample Usage
1) Instantiate the class object
```
  parameter_sets = [(2, 2), (3, 2), (5, 1), (9, 0.5), (7, 1), (2, 1), (3, 1), (4, 1), (3, 3)]

  df = pd.read_csv("dataset.csv") #Replace with your dataset
  # Assumes df is already processed - contains only X and y, and is already label encoded and scaled appropriately
  target_col = "class"
  categorical_col_names = ["cat_col_1","cat_col_2","cat_col_3"] #Leave empty if no categorical columns
  balancer = GammaBalancerBinary(df, target_col, majority_class=1, minority_class=0, categorical_col_names=categorical_col_names)
```
2) Find the best shape and scale hyperparameters
```
  # Use the validation set to find the best shape and scale params
  best_shape, best_scale = balancer.find_best_params(X_train, X_val, y_train, y_val, parameter_sets=parameter_sets)
```

3) Oversample the minority class based on the shape and scale params found
```
  df_train = pd.concat([X_train,y_train],axis=1)
  df_train_oversampled = balancer.oversample_minority_class(df_train, best_shape, best_scale)
```

4) Use the oversampled dataset as needed - Eg: To train a model
```
  # Split the new oversampled train dataset back into xtrain and ytrain
  X_train_oversampled = df_train_oversampled.drop(target_col, axis=1)
  y_train_oversampled = df_train_oversampled[target_col]

  # Use random forest classifier:
  model = RandomForestClassifier()
  model.fit(X_train_oversampled, y_train_oversampled)

  # Predict and evaluate on the test set
  y_pred = model.predict(X_test)
```
------------------------------------------
# Citation:
```
TBD Once paper releases
```
