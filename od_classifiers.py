from sklearn.grid_search import GridSearchCV
from sklearn.externals.six import StringIO
import pylab as pl
import numpy as np
import pydot
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,
recall_score, f1_score


# Classifier/parameter candidates:

clfs_1 = [
    GaussianNB(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier()
]

par_1 = [
    None,
    {'min_samples_split': [2, 4, 6, 16], 'criterion': ['gini', 'entropy']},
    {'n_estimators': [50, 100, 150, 200, 250, 300]},
    {'n_estimators': [10, 20, 30, 40, 50, 60]},
    {'n_neighbors': [5, 10, 15, 20, 25, 30], 'metric': ['minkowski']}
]

clfs_2 = [SVC()]

par_2 = [{'kernel': ['linear', 'rbf'], 'C': [1., 10., 100., 1000.]}]


# Loading, Cleaning, and Joining Datasets:

od = pandas.read_csv(r"May_2013_Online_Dating_csv.csv")  # loading 2013 data

# features of interest

feature_list = ['sex', 'age', 'race', 'hisp', 'educ2', 'inc', 'state',
                'intuse', 'date3a', 'date3b', 'par', 'mar', 'mar2', 'lgbt']

# to be fed to bucket_maker (to create marriage feature)

buckets = {'mar_bucket': {1: {'mar': [1]}, 2: {'mar': [2, 3, 4, 5, 6]}}}


def bucket_maker(buckets, data):    # creates bucket features
    for e in buckets:
        data[e] = 0
        for f in buckets[e]:
            for g in buckets[e][f]:
                for h in buckets[e][f][g]:
                    data[e][data[g] == h] = f


def clean_2013(data, features):    # converts "don't know" or "refuse" to 0's
    for e in features + ['date1a']:
        if e in ['age', 'educ2', 'inc', 'state']:
            data[e].replace([98, 99], [0, 0], inplace=True)
        elif e in ['date1a', 'intuse', 'mar2']:
            data[e].replace(['1', '2', '8', '9', ' '], [1, 2, 0, 0, 0],
                            inplace=True)
        else:
            data[e].replace([8, 9], [0, 0], inplace=True)

clean_2013(od, feature_list)

bucket_maker(buckets, od)

# Loading 2005 data (doesn't include 'lgbt' field)

od_2005 = pandas.read_csv(r"September.2005_csv.csv")

# Renaming columns to match 2013 data

od_2005.rename(columns={'date01a': 'date3a', 'date01c': 'date3b',
                        'educ': 'educ2', 'q6a': 'intuse'}, inplace=True)

# creating 'date1a' (target) column

buckets_2005 = {'date1a': {1: {'activ33': ['1', '2']}, 2: {'activ33': ['3']}}}


def clean_2005(data, features):
    for e in features:
        if e in ['age', 'state']:
            data[e].replace([98, 99], [0, 0], inplace=True)
        elif e in ['mar']:
            data[e].replace([8, 9], [0, 0], inplace=True)
        elif e in ['mar2', 'date3b']:
            data[e].replace(['1', '2', '9', ' '], [1, 2, 0, 0], inplace=True)
        else:
            data[e].replace([9], [0], inplace=True)

clean_2005(od_2005, feature_list[0:13])

bucket_maker(buckets, od_2005)

bucket_maker(buckets_2005, od_2005)

od = od.append(od_2005)   # joining cleaned datasets

# appending leads to Na's which need to be replaced with 0's

od.fillna(0, inplace=True)

# appending datasets led to duplicate indices: index must be reset to give
# each row unique value

od.index = [e for e in range(0, len(od))]


# Imputing:

impute = {  # to be fed to imputer
    'date3a': {'clf': [AdaBoostClassifier()], 'nfeats': 12},
    'mar_bucket': {'clf': [RandomForestClassifier()], 'nfeats': 1}}

# redefining feature list to exclude 'mar' and include 'mar_bucket'

feature_list = [x for x in feature_list if x != 'mar'] + ['mar_bucket']


def imputer(data, impute):   # replaces 0's with classifier's predictions
    imputed = []
    for e in impute:
        col = e
        new = data[data[e] != 0]
        counts = new[e].value_counts()
        if len(counts) == 2:
            max_index, min_index = counts.argmax(), counts.argmin()
            g_1, g_2 = new[new[e] == min_index], new[new[e] == max_index]
            g_2 = g_2.loc[np.random.choice(g_2.index, int(len(g_1)),
                                           replace=False)]
            new = g_1.append(g_2)
        indices = data[data[e] == 0].index
        sub_features = [x for x in feature_list if x != e]
        X = new[sub_features]
        X = X.as_matrix()
        Y = new[e].as_matrix()
        selector = SelectKBest(f_classif, k=impute[e]['nfeats'])
        X = selector.fit_transform(X, Y)
        c = impute[e]['clf'][0]
        ind = [clfs_1.index(e) for e in clfs_1
               if type(e).__name__ == type(c).__name__][0]
        par = par_1[ind]
        if par:
            clf = GridSearchCV(c, par, cv=10)
        else:
            clf = c
        clf.fit(X, Y)
        for i in indices:
            pred = clf.predict(
                selector.transform(data[sub_features].ix[i].as_matrix()))[0]
            data.loc[i, col] = pred
        imputed.append([col, data[col]])
    return imputed

for e in imputer(od, impute):  # calling imputer / replacing columns
    od[e[0]] = e[1]

# including online dating app. users in online dater group

od.loc[od['date2a'] == '1', 'date1a'] = 1

# if you know someone who met partner o.d., you know someone who has tried o.d.

od.loc[od['date3b'] == 1, 'date3a'] = 1


# Alleviating Skewness:

# splitting data into online daters and non-online daters

group_1, group_2 = od[od.date1a == 1], od[od.date1a == 2]

# limiting number of non-online daters (to alleviate skewness) to 269
# (number of online daters)

group_2 = group_2.loc[np.random.choice(group_2.index, 269, replace=False)]

od = group_1.append(group_2)   # rejoining two groups


# Creating Features/Target numpy arrays:

features = od[['date3a', 'mar_bucket']]

features = features.as_matrix()

target = od['date1a'].as_matrix()


# Rescaling Features (not necessary for decision tree):

scaler = MinMaxScaler()

features = scaler.fit_transform(features.astype(float))


# Building Decision Tree:

clf = DecisionTreeClassifier()

clf.fit(features, target)


# Visualizing Tree:

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data, feature_names=['date3a', 'mar_bucket'])

graph = pydot.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("od_clf.pdf")


# Functions used to evaluate models (modified from
# http://sujitpal.blogspot.com/2013/05/feature-selection-with-scikit-learn.html):

# performs cross validation and returns metrics for given model


def evaluate(features, target, c, nfeats, clfname, par, pca_on=False):
    a, f, p, r = 0, 0, 0, 0
    cm = np.matrix([[0, 0], [0, 0]])
    kf = KFold(len(features), n_folds=10, shuffle=True)
    count = 0
    for train_indices, test_indices in kf:
        count += 1
        features_train = features[train_indices]
        features_test = features[test_indices]
        labels_train = target[train_indices]
        labels_test = target[test_indices]
        if pca_on:
            pca = PCA(n_components=nfeats)
            features_train = pca.fit_transform(features_train)
            features_test = pca.transform(features_test)
        else:
            selector = SelectKBest(f_classif, k=nfeats)
            features_train = selector.fit_transform(features_train,
                                                    labels_train)
            features_test = selector.transform(features_test)
        if par:
            clf = GridSearchCV(c, par, cv=3)
        else:
            clf = c
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        a_score = accuracy_score(labels_test, pred)
        f_score = f1_score(labels_test, pred)
        p_score = precision_score(labels_test, pred)
        r_score = recall_score(labels_test, pred)
        c_matrix = confusion_matrix(labels_test, pred)
        a, f, p, r, cm = a + a_score, f + f_score, p + p_score, r + r_score,
        cm + c_matrix
    return a/10, f/10, p/10, r/10

# plots metric vs. # of features for each model


def plot(scores, xvals, legends, ylab):
    pl.figure()
    pl.title(ylab + ' vs Number of Features for Different Classifiers')
    for e in range(0, len(legends)):
        pl.plot(xvals, scores[e, :], lw=2, label=legends[e])
    pl.legend(loc=3, prop={'size': 10})
    pl.xlabel('Number of Features')
    pl.ylabel(ylab)
    pl.show()

# coordinates previous two functions


def main(features, target, nFeatures, clfs, par):
    clfnames = [type(e).__name__ for e in clfs]
    dim = (len(clfs), len(nFeatures))
    accuracies, f1s, precisions, recalls = np.zeros(dim), np.zeros(dim),
    np.zeros(dim), np.zeros(dim)
    for j in range(0, len(nFeatures)):
        for i in range(0, len(clfs)):
            accuracies[i, j], f1s[i, j], precisions[i, j], recalls[i, j] =
            evaluate(features, target, clfs[i], nFeatures[j], clfnames[i],
                     par[i], pca_on=False)
    scores = {'Accuracy': accuracies, 'F': f1s, 'Precision': precisions,
              'Recall': recalls}
    for name in scores:
        plot(scores[name], nFeatures, clfnames, name)
