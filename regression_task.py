import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures    # to convert the original features into their higher order terms
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from category_encoders import OneHotEncoder, BinaryEncoder, SumEncoder, PolynomialEncoder
import seaborn as sns


def encode_one_hot(x):
    """Encodes the Dataframe with the OneHotEncoder
    :param x: X Dataframe which needs to be Encoded, Fitted and Transformed"""
    category_features = x.select_dtypes(include=['object']).columns.tolist()
    print("\nCategorical Features : \n", category_features)
    encoder = OneHotEncoder(cols=category_features)
    encoder.fit(x)
    x = encoder.transform(x)
    print("\nEncoded bit rate x_train : \n", x)
    return x

def impute_simple(x, strategy='mean'):
    """Imputes the Dataframe with the SimpleImputer
    :param x: X Dataframe which needs to be Imputed, Fitted and Transformed
    :param strategy: The strategy by which the dataframe needs to be imputed"""

    imputer = SimpleImputer(strategy='mean')
    imputer.fit(x)
    x = pd.DataFrame(imputer.transform(x), columns=x.columns)
    print("\nImputed Bit Rate : \n", x)
    return x

def scale_std(x):
    """Scales the Dataframe with the RobustScaler
    :param x: X Dataframe which needs to be Scaled, Fitted and Transformed"""

    scaler = RobustScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=x.columns)
    print("\nScaled Bit Rate : \n", x)
    return x

def data_preprocessing(x):
    """Data preprocessing - Encoding, Imputing, Feature Selection, Scaling of the Dataframe
    :param x: X Dataframe which needs to be Encoded, Imputed,
     Scaled, Fitted and Transformed"""

    # Encoding the data with the One Hot Encoder x_train
    x = encode_one_hot(x)

    # Imputing the data with the mean x_train
    x = impute_simple(x, strategy='mean')

    # Feature Selection
    print("Shape before Feature Selection : ", x.shape)
    x = x.replace(0, np.nan)
    x = x.dropna(axis=1, thresh=0.05*len(x))
    print("Removed 95% zero columns : \n", x)
    print("shape after Feature Selection : ", x.shape)
    x = x.replace(np.nan, 0)

    # Scaling the data with Robust Scaler x_train
    x = scale_std(x)
    return x

def pair_plot(x, suptitle_label='Pair Plot of X data'):
    """Plots a Pair Plot from a Dataframe
    :param x: X Dataframe which needs to be pair-plotted
    :param suptitle_label: It's a label for the plot"""

    plt.figure()
    sns.pairplot(pd.DataFrame(x))
    plt.suptitle(suptitle_label)
    plt.show()
    return None

def print_metrics(y_true, y_predict):
    """Prints the Metrics such as MSE, MAE, RMSE, R2 Score of the
    true and predicted values of a particular model
    :param y_true: True target dataframe
    :param y_predict: Predicted target dataframe"""

    print('Mean Squared Error       :', metrics.mean_squared_error(y_true, y_predict))
    print('Mean Absolute Error      :', metrics.mean_absolute_error(y_true, y_predict))
    print('Root Mean Squared Error  :', metrics.mean_squared_error(y_true, y_predict, squared=False))
    print("R2 Score                 : ", r2_score(y_true=y_true, y_pred=y_predict))
    return None

def remove_outliers(x, coln):
    q1 = x[coln].quantile(0.25)
    q3 = x[coln].quantile(0.75)
    iqr = q3-q1     # Inter quartile range
    thresh_low = q1-1.5*iqr
    thresh_high = q3+1.5*iqr
    df_out = x.loc[(x[coln] > thresh_low) & (x[coln] < thresh_high)]
    return df_out

if __name__ == '__main__':

    # Read the train and test dataframes from csv files
    bit_rate = pd.read_csv("bitrate_prediction/bitrate_train.csv")
    bit_rate_test = pd.read_csv("bitrate_prediction/bitrate_test.csv")

    # Plot the correlation Matrix of the Bit Rate train
    f = plt.figure(figsize=(19, 15))
    plt.matshow(bit_rate.corr(), fignum=f.number)
    plt.xticks(range(bit_rate.select_dtypes(['number']).shape[1]), bit_rate.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(bit_rate.select_dtypes(['number']).shape[1]), bit_rate.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix of Bit Rate Train', fontsize=16)
    plt.show()

    # Profile Report in .html format
    # ProfileReport(bit_rate).to_file("Bit-Rate-Profile-Report.html")

    # Describe the bit rate train
    print(bit_rate.describe())

    # Remove Outliers from Train set
    features = list()
    for col in bit_rate.drop(['target'], axis=1).columns:
        features.append(col)
    print("All the features in bitrate dataframe : ", features)

    # Outliers
    bit_rate.boxplot(figsize=(20, 16))

    print("bit rate shape before removing outliers: ", bit_rate.shape)
    bit_rate = remove_outliers(bit_rate, features[1])
    bit_rate = remove_outliers(bit_rate, features[2])
    bit_rate = remove_outliers(bit_rate, features[3])
    print("bit rate shape after removing outliers: ", bit_rate.shape)

    bit_rate.boxplot(figsize=(20, 16))

    # Remove Outliers from Test set
    features = list()
    for col in bit_rate_test.drop(['target'], axis=1).columns:
        features.append(col)
    print("All the features in bitrate test dataframe : ", features)

    # Outliers
    bit_rate_test.boxplot(figsize=(20, 16))

    print("bit rate test shape before removing outliers: ", bit_rate_test.shape)
    bit_rate_test = remove_outliers(bit_rate_test, features[1])
    bit_rate_test = remove_outliers(bit_rate_test, features[2])
    bit_rate_test = remove_outliers(bit_rate_test, features[3])
    print("bit rate test shape after removing outliers: ", bit_rate_test.shape)

    bit_rate_test.boxplot(figsize=(20, 16))

    # Splitting the 'target' column from the x_train, x_test
    # and storing it in y_train, y_test respectively
    # We Get the x_train, y_train and x_test, y_test data
    x_train = bit_rate.drop(['target'], axis=1)
    y_train = bit_rate.loc[:, 'target']
    x_test = bit_rate_test.drop(columns=['target'])  # We can also use the columns arg to avoid writing axis = 1
    y_test = bit_rate_test.loc[:, 'target']

    # Data preprocessing - Encoding, Imputing, Feature Selection, Scaling
    x_train = data_preprocessing(x_train)
    x_test = data_preprocessing(x_test)

    # Selected Features after data preprocessing stage
    selected_features = list()
    for col in x_train.columns:
        selected_features.append(col)
    print("\nSelected Features : ", selected_features)

    # Pair Plot of x_train after preprocessing of the data
    # This takes latest 4 minutes to execute
    # pair_plot(x_train, suptitle_label="Pair Plot of x_train data")

    # Split the x_train to two parts x_val and x_train
    x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=0)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=0)
    print(f"\nShape of x_train : {x_train.shape}\nShape of x_val : {x_val.shape}")
    print(f"Shape of y_train : {y_train.shape}\nShape of y_val : {y_val.shape}")

    # PCA Visualization of the Complete Dataframe x_train in 2dimensional graphs
    print("\nShape of x_train before PCA", x_train.shape)
    dimes_reducer = PCA(n_components=1)
    x_train_reduced = dimes_reducer.fit_transform(x_train)
    f = plt.figure(figsize=(19, 15))
    plt.scatter(x_train_reduced[:, 0], y_train, marker='o')    # x_train_reduced[:, 1]
    plt.suptitle("PCA Visualization of Bit-Rate x_train data: ")
    plt.xlabel("x_train")
    plt.ylabel("y_train")
    plt.show()
    print("\nShape of x_train after PCA", x_train_reduced.shape)

    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    print(f"\nLinear intercept : {linear_model.intercept_}")
    print(f"Linear Coefficient : {linear_model.coef_}")

    # Predict the x_test and x_val with the Linear Model
    y_pred_test = linear_model.predict(x_test)
    y_pred_val = linear_model.predict(x_val)

    # Performance of the Linear Regression Model for the Bitrate Prediction
    # print(pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}))
    print("\nMetrics for Linear Model for Test set")
    print_metrics(y_true=y_test, y_predict=linear_model.predict(x_test))
    print("\nMetrics for Linear Model for Validation set")
    print_metrics(y_true=y_val, y_predict=linear_model.predict(x_val))
    print("\nLinear Model Score for Training set : ", linear_model.score(x_test, y_test))
    print("Linear Model Score for Validation set : ", linear_model.score(x_val, y_val))

    # Check shapes for x_test and y_test
    print(x_test.shape, y_test.shape)

    # Polynomial Regression Model
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = poly.fit_transform(x_train)
    poly_regress_model = LinearRegression()
    poly_regress_model.fit(poly_features, y_train)
    y_pred_poly = poly_regress_model.predict(poly_features)

    # Predict the x_test and x_val with the Polynomial Model
    poly_features_test = poly.fit_transform(x_test)
    y_pred_poly_test = poly_regress_model.predict(poly_features_test)

    poly_features_val = poly.fit_transform(x_val)
    y_pred_poly_val = poly_regress_model.predict(poly_features_val)

    # Plotting the Polynomial Regression graphs
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    axes[0].scatter(x_train['fps_mean'], y_train)
    axes[1].scatter(x_train['fps_std'], y_train)
    axes[2].scatter(x_train['rtt_mean'], y_train)
    axes[0].set_title('fps_mean')
    axes[1].set_title('fps_std')
    axes[2].set_title('rtt_mean')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    axes[0].scatter(x_train['bitrate_mean'], y_train)
    axes[1].scatter(x_train['bitrate_std'], y_train)
    axes[2].scatter(x_train['rtt_std'], y_train)
    axes[0].set_title('bitrate_mean')
    axes[1].set_title('bitrate_std')
    axes[2].set_title('rtt_std')
    plt.show()

    # Polynomial Model with 2nd Degree Plot
    f = plt.figure(figsize=(19, 15))
    plt.scatter(x_train_reduced[:, 0], y_train, marker='o')  # x_train_reduced[:, 1]
    plt.plot(x_train, y_pred_poly, c='blue')
    plt.suptitle("Polynomial Model with 2nd Degree")
    plt.xlabel("x_train")
    plt.ylabel("y_train")
    plt.show()

    # Performance of the Polynomial Regression Model for the Bitrate Prediction
    print(y_test.shape, y_pred_poly_test.shape)
    print("\nMetrics for Polynomial Model for Test set")
    print_metrics(y_true=y_test, y_predict=y_pred_poly_test)
    print("\nMetrics for Polynomial Model for Validation set")
    print_metrics(y_true=y_val, y_predict=y_pred_poly_val)

    # Lasso and Ridge
    lasso = Lasso()
    lasso.fit(x_train, y_train)
    ridge = Ridge()
    ridge.fit(x_train, y_train)
    print('\nLasso coef', lasso.coef_)  # it produces sparse models
    print('\nRidge coef', ridge.coef_)

    # Lasso Model
    # Calculating the best Alpha for Lasso Model
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    mse_s = []
    for alpha in alphas:
        lasso_model = Lasso(alpha=alpha).fit(x_train, y_train)  # change lasso to ridge and you will get some other value
        mse = mean_squared_error(lasso_model.predict(x_val), y_val)
        mse_s.append(mse)
    plt.plot(alphas, mse_s)
    plt.title("Lasso alpha value selection")
    plt.xlabel("alpha")
    plt.ylabel("Mean squared error")
    plt.show()

    best_alpha = alphas[np.argmin(mse_s)]
    print("Best value of alpha for Lasso Model is : ", best_alpha)

    lasso_model = Lasso(alpha=best_alpha)
    lasso_model.fit(x_train, y_train)
    y_lasso_pred_test = lasso_model.predict(x_test)
    y_lasso_pred_val = lasso_model.predict(x_val)

    # Performance of the Ridge Model for the Bitrate Prediction
    print("\nMetrics for Lasso Model for Test set")
    print_metrics(y_true=y_test, y_predict=y_lasso_pred_test)
    print("\nMetrics for Lasso Model for Validation set")
    print_metrics(y_true=y_val, y_predict=y_lasso_pred_val)

    # Ridge Model
    # Calculating the best Alpha for Ridge Model
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    mse_s = []
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha).fit(x_train, y_train)  # change lasso to ridge and you will get some other value
        mse = mean_squared_error(ridge_model.predict(x_val), y_val)
        mse_s.append(mse)
    plt.plot(alphas, mse_s)
    plt.title("Ridge alpha value selection")
    plt.xlabel("alpha")
    plt.ylabel("Mean squared error")
    plt.show()

    best_alpha = alphas[np.argmin(mse_s)]
    print("Best value of alpha for Ridge Model is : ", best_alpha)

    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(x_train, y_train)
    y_ridge_pred_test = ridge_model.predict(x_test)
    y_ridge_pred_val = ridge_model.predict(x_val)

    # Performance of the Ridge Model for the Bitrate Prediction
    print("\nMetrics for Ridge Model for Test set")
    print_metrics(y_true=y_test, y_predict=y_ridge_pred_test)
    print("\nMetrics for Ridge Model for Validation set")
    print_metrics(y_true=y_val, y_predict=y_ridge_pred_val)

    print("Ridge Model Score for Validation set : ", ridge.score(x_val, y_val))

    # print(y_pred)

    # The accuracy_score, precision, recall, f1 ... are for Classification tasks
    # print(metrics.accuracy_score(list(y_test), list(y_pred)))
    # print(metrics.precision_score(y_test, y_pred))
    # print(metrics.recall_score(y_test, y_pred))
