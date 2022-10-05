import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.linear_model import Lasso, Ridge
from pandas_profiling import ProfileReport
from category_encoders import OneHotEncoder, BinaryEncoder, SumEncoder, PolynomialEncoder
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

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

def remove_outliers(x, coln):
    """Detects and removes the outliers from the dataframe
    :param x: Dataframe
    :param coln: Feature/Column name from the dataframe"""
    q1 = x[coln].quantile(0.25)
    q3 = x[coln].quantile(0.75)
    iqr = q3-q1     # Inter quartile range
    thresh_low = q1-1.5*iqr
    thresh_high = q3+1.5*iqr
    df_out = x.loc[(x[coln] > thresh_low) & (x[coln] < thresh_high)]
    return df_out

def print_metrics(y_true, y_predict):
    """Prints the Metrics such as Accuracy, Precision, Recall, F1 Score of the
    true and predicted values of a particular classifier model
    :param y_true: True target dataframe
    :param y_predict: Predicted target dataframe"""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_predict, average='macro')
    print("Accuracy, Precision, Recall and F1_score are: ")
    print("Precision : ", p)
    print("Recall : ", r)
    print("F1 score : ", f1)
    a = accuracy_score(y_true, y_predict)
    print("Accuracy : ", a)
    print("\nClassification Report:")
    target_names = [ 'Class 0', 'Class 1',]
    print(classification_report(y_true, y_predict, target_names=target_names))
    return None

if __name__ == '__main__':

    # Read the train and test data from csv files
    stream_quality = pd.read_csv("stream_quality_data/train_data.csv")
    stream_quality_test = pd.read_csv("stream_quality_data/test_data.csv")

    # Plot the correlation Matrix of the Stream Quality train
    f = plt.figure(figsize=(19, 15))
    plt.matshow(stream_quality.corr(), fignum=f.number)
    plt.xticks(range(stream_quality.select_dtypes(['number']).shape[1]), stream_quality.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(stream_quality.select_dtypes(['number']).shape[1]), stream_quality.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix of Stream Quality Train', fontsize=16)
    plt.show()

    # Profile Report in .html format
    # ProfileReport(stream_quality).to_file("Stream-Quality-Profile-Report.html")

    # Describe the Stream Quality train
    print(stream_quality.describe())

    # Remove Outliers from Train set
    features = list()
    for col in stream_quality.drop(['stream_quality'], axis=1).columns:
        features.append(col)
    print("All the features in stream quality dataframe : ", features)

    # Outliers
    stream_quality.boxplot(figsize=(20, 16))

    print("stream quality shape before removing outliers: ", stream_quality.shape)
    stream_quality = remove_outliers(stream_quality, features[1])
    stream_quality = remove_outliers(stream_quality, features[3])
    stream_quality = remove_outliers(stream_quality, features[4])
    print("stream quality shape after removing outliers: ", stream_quality.shape)

    stream_quality.boxplot(figsize=(20, 16))

    # Remove Outliers from Test set
    features = list()
    for col in stream_quality_test.drop(['stream_quality'], axis=1).columns:
        features.append(col)
    print("All the features in stream quality test dataframe : ", features)

    # Outliers
    stream_quality_test.boxplot(figsize=(20, 16))

    print("stream quality test shape before removing outliers: ", stream_quality_test.shape)
    stream_quality_test = remove_outliers(stream_quality_test, features[1])
    stream_quality_test = remove_outliers(stream_quality_test, features[3])
    stream_quality_test = remove_outliers(stream_quality_test, features[4])
    print("stream quality test shape after removing outliers: ", stream_quality_test.shape)

    stream_quality_test.boxplot(figsize=(20, 16))

    # Splitting the 'stream_quality' column from the train_data
    # and storing it in y_train, y_test respectively
    # We Get the x_train, y_train and x_test, y_test data
    x_train = stream_quality.drop(['stream_quality'], axis=1)
    y_train = stream_quality.loc[:, 'stream_quality']
    x_test = stream_quality_test.drop(columns=['stream_quality'])  # Can use the columns arg to avoid writing axis = 1
    y_test = stream_quality_test.loc[:, 'stream_quality']

    # Data preprocessing - Encoding, Imputing, Feature Selection, Scaling
    x_train = data_preprocessing(x_train)
    x_test = data_preprocessing(x_test)

    # Selected Features after data preprocessing stage
    selected_features = list()
    for col in x_train.columns:
        selected_features.append(col)
    print("\nSelected Features : ", selected_features)

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

    # Logistic Regression Model with L2 Regularization
    X = x_train.values
    logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500000, random_state=0).fit(x_train, y_train)

    y_pred = logistic_model.predict(x_test)
    y_pred_prob = logistic_model.predict_proba(x_test)

    print("\nPerformance Metrics of Logistic Regression with L2 Regularization:")
    print_metrics(y_true=y_test, y_predict=y_pred)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y_pred_prob = logistic_model.predict_proba(x_test)
    vals = [[], [], []]
    for i in thresholds:
        y_test_pred_thr = np.where(y_pred_prob[:, 1] > i, 1, 0)
        vals[0].append(metrics.accuracy_score(y_test, y_test_pred_thr))
        vals[1].append(metrics.precision_score(y_test, y_test_pred_thr, zero_division=1))
        vals[2].append(metrics.recall_score(y_test, y_test_pred_thr))

    plt.plot(thresholds, vals[0], label='Accuracy')
    plt.plot(thresholds, vals[1], label='Precision')
    plt.plot(thresholds, vals[2], label='Recall')
    plt.title('Logistic regression with L2 - threshold selection')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

    # Logistic Regression with Over sampling
    o_sampler = RandomOverSampler(sampling_strategy=0.5)
    x_over, y_over = o_sampler.fit_resample(x_train, y_train)
    print(y_train.value_counts())
    print(y_over.value_counts())

    logistic_samp_model = LogisticRegression().fit(x_over.values, y_over)
    pred_over = logistic_samp_model.predict(x_test)
    print("\nPerformance Metrics of Logistic Regression with Over Sampling:")
    print_metrics(y_true=y_test, y_predict=pred_over)

    # DecisionTreeClassifier
    decision_tree_model = DecisionTreeClassifier().fit(x_train, y_train)
    y_pred_dt = decision_tree_model.predict(x_test)
    print("\nPerformance Metrics of Decision Tree Classifier:")
    print_metrics(y_true=y_test, y_predict=y_pred_dt)

    # Support Vector Machine Classifier
    svm_model = SVC(kernel='linear').fit(x_train, y_train)
    y_pred_svm = svm_model.predict(x_test)
    print("\nPerformance Metrics of Support Vector Machine Classifier")
    print_metrics(y_true=y_test, y_predict=y_pred_svm)
