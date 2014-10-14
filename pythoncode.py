import pandas as pd
import scipy.stats
from sklearn import ensemble, linear_model, neural_network, cross_validation, feature_selection#, pipeline, preprocessing

if __name__ == "__main__":
  train = "train.csv"
  test = "test.csv"
  gbmsubmission = "gbmsubmission.csv"
  #rfsubmission = "rfsubmission.csv"
  #gbcsubmission = "gbcsubmission.csv"
  #sgdsubmission = "sgdsubmission.csv"
  #adasubmission = "adasubmission.csv"
  #rbmsubmission = "rbmsubmission.csv"

  df_train = pd.read_csv(train)
  df_test = pd.read_csv(test)


  X_train, X_test = cross_validation.train_test_split(df_train,test_size=.25)
  X_train = pd.DataFrame(X_train)
  X_test = pd.DataFrame(X_test)
  feature_cols = [col for col in X_train.columns if col not in [0,55]]#range(1,11)]
  train = X_train[feature_cols]
  y = X_train[55]
  Y_test = X_test[55]
  test = X_test[feature_cols]

  #test_ids = df_test['Id']
  #X_train_scaled = scaler.fit_transform(X=X_train.astype(np.float))
  #min_max_scaler = preprocessing.MinMaxScaler()
  #X_train_minmax = min_max_scaler.fit_transform(X_train)
  #X_test_minmax = min_max_scaler.fit_transform(X_test)
  
  

  gbm = ensemble.GradientBoostingClassifier(n_estimators=50,max_depth=20,learning_rate=.2)
  rf = ensemble.RandomForestClassifier(n_estimators = 1500, max_features=None, n_jobs = -1)
  gbc = ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000)
  sgd = linear_model.SGDClassifier(n_iter=50, n_jobs=5)
  ada = ensemble.AdaBoostClassifier(n_estimators=500)
  rbm = neural_network.BernoulliRBM(n_components=400,n_iter=25)
  logistic = linear_model.LogisticRegression()
  #classifier = pipeline.Pipeline([("rbm", rbm), ("logistic", logistic)])
  gbm.fit(X_train, y)
  rf.fit(X_train, y)
  gbc.fit(X_train, y)
  sgd.fit(X_train, y)
  ada.fit(X_train, y)
  #classifier.fit(X_train_minmax, y)

  
  '''with open(rfsubmission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(rf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))'''

  '''with open(gbcsubmission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(gbc.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))

  with open(sgdsubmission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(sgd.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))'''

  '''with open(adasubmission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(ada.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))'''

  '''with open(rbmsubmission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(classifier.predict(X_test_minmax))):
      outfile.write("%s,%s\n"%(test_ids[e],val))'''
  '''with open(gbmsubmission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(gbm.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))'''