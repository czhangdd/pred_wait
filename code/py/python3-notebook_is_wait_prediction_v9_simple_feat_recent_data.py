# Python Notebook - is_wait_prediction_v9_simple_feat_recent_data

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime, timedelta
from sklearn.feature_selection import RFECV

def eval_model(y_test, y_pred, y_pred_proba):
  acc_score = accuracy_score(y_test, y_pred)
  logloss = log_loss(y_test, y_pred_proba)
  logloss_baseline = log_loss(y_test, np.zeros(len(y_test)))
  rocauc = roc_auc_score(y_test, y_pred_proba[:, 1])
  conf_mat = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = conf_mat.ravel()

  print('true pct in testing data of no-wait', 1-y_test.mean(), 'vs wait', y_test.mean(), )
  print('acc', acc_score)
  print('log loss pred vs baseline', logloss, logloss_baseline)
  print('roc auc', rocauc)
  print('conf matrix (percentage of all data)\n', conf_mat/len(y_test))
  print("absolute values of tn", tn, "fp", fp, "fn", fn, "tp", tp)

def create_store_level_hist_feat(df):
  d = {'wait_before_ready_time': ['mean', 'min', 'max'], 'd2r_duration': ['mean', 'min', 'max'], 'pred_horizon': ['mean', 'min', 'max']}
  df_agg = df.groupby(['store_id'], as_index=False).agg(d)
  
  df_agg.columns = ["_".join(x) for x in df_agg.columns.ravel()]
  df_agg.rename(columns={'store_id_':'store_id'}, inplace=True)
  
  df_hist = pd.merge(df, df_agg, on=['store_id'], how='left')
  return df_hist 

wait_thr = 5 * 60
long_wait_thr = 10 * 60
delay_thr = 0

# df = datasets['4_feat_filter']
df = datasets['6_wait_geo']
df.columns = map(str.lower, df.columns)
raw_data_size = df.shape
print("raw_data_size", raw_data_size)
print(df.head(5))

f = 'pred_horizon'

df_clean = df.copy()
df_clean = df_clean[df_clean['flf'] < 5]
df_clean = df_clean[(df_clean['pred_horizon'] < 60 * 60) & (df_clean['pred_horizon'] >= 0)]

print('after remove pred_horizon and flf, shape', df_clean.shape)

df_clean[f].min(), df_clean[f].max()

## Outlier removal
print(">>> check NAs")
print(df.isna().sum() / len(df))

print(">>> remove data with invalid lables")
df_clean = df_clean[df_clean['wait_before_ready_time'].notna()]
df_clean.reindex()
print('>>> after dropping NAs in wait_time (label)', df_clean.shape, 1-df_clean.shape[0]/raw_data_size[0])

print('>>> drop NA with 0')
df_clean.dropna(inplace=True)
# df_clean = df_clean.fillna(0)

print('>>> after dropping NAs in other columns', df_clean.shape, 1-df_clean.shape[0]/raw_data_size[0])

df = df_clean
print(">>> check if na?", df.isnull().values.any())

df = create_store_level_hist_feat(df)
print(">>> adding agg features")
print(df.columns)

# add lateness labels
df['label'] = "ontime"
df.loc[df['wait_before_ready_time'] > wait_thr, 'label'] = 'early'
df.loc[df['wait_before_ready_time'] < -delay_thr, 'label'] = 'late'


df['label_int'] = 0
df.loc[df['label'] == 'early', 'label_int'] = 1
df.loc[df['label'] == 'ontime', 'label_int'] = 0
df.loc[df['label'] == 'late', 'label_int'] = -1

assert (df['label_int'].value_counts(normalize=True).values ==  df['label'].value_counts(normalize=True).values).all(), "error in converting str label to int label"

print('Percentage of labels')
print(df['label'].value_counts(normalize=True))

# manually select top features
feat_hist_agg = ['wait_before_ready_time_mean', 'd2r_duration_mean', 'wait_before_ready_time_max',\
                 'd2r_duration_min', 'pred_horizon_mean', 'wait_before_ready_time_min' , 'avg_num_assigns']
feat_real_time = ['tip', 'flf', 'pred_horizon', 'num_busy', 'num_nearby_idle']

cols = feat_hist_agg + feat_real_time
for f in cols:
  df[f] = df[f].astype('float32')

df['created_at'] = pd.to_datetime(df['created_at'])
date_train_end = df['created_at'].dt.date.max() - timedelta(weeks=1)

train_index = df[df['created_at'] <= date_train_end].index
test_index = df[df['created_at'] > date_train_end].index
assert len(train_index.intersection(test_index)) != [], 'data leakage'
print('train data size', len(train_index), 'date', df.loc[train_index, 'created_at'].min(), df.loc[train_index, 'created_at'].max())
print('test data size', len(test_index), 'date', df.loc[test_index, 'created_at'].min(), df.loc[test_index, 'created_at'].max())

X_train = df.loc[train_index, cols]
y_train = df.loc[train_index, 'label']
y_train_int = df.loc[train_index, 'label_int']

X_test = df.loc[test_index, cols]
y_test = df.loc[test_index, 'label']
y_test_int = df.loc[test_index, 'label_int']

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = Normalizer()
scaler.fit(X_train)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[X_train.columns] = scaler.transform(X_train)
X_test_scaled[X_test.columns] = scaler.transform(X_test)

for i in range(X_train_scaled.shape[1]):
  assert X_train_scaled.values[:, i].max() > 0.999, 'check if all scaled features max == 1'

sorted_corr_col = df[cols+['label_int']].corr()['label_int'].abs().sort_values(ascending=False).index

df[sorted_corr_col].corr()['label_int'][1:].plot.bar(rot=90, title="raw corr with label")

estimator = LogisticRegression()

selector = RFECV(estimator, step=0.3, cv=5)
selector = selector.fit(X_train_scaled, y_train.values)
selector.support_

print(selector.support_)

# Create interaction term (not polynomial features)
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.regression import linear_model

interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_scaled_selected = X_train_scaled[feat_real_time]
X_test_scaled_selected = X_test_scaled[feat_real_time]


X_train_scaled_selected_inter = interaction.fit_transform(X_train_scaled_selected)

X_train_scaled_selected_inter = interaction.fit_transform(X_test_scaled_selected)


# X_train_scaled_selected.shape, X_train_scaled_selected_inter.shape

# print(feat_real_time)


X_train_scaled_selected_inter.shape, y_train.values.shape

interaction_model = linear_model.OLS(y_train.values, X_train_scaled_selected_inter).fit()
# print(feat_real_time)
# print(interaction_model)
interaction_model.pvalues[interaction_model.pvalues < 0.05]
# X_train_scaled_selected_inter.shape, X_train_scaled.shape

interaction_model.summary()


inter_items.reindex(X_train_scaled.index)

tmp = X_train_scaled_selected_inter[:, [9, 12]]
# inter_items = pd.DataFrame(tmp, columns=['inter'+ str(i) for i in range(tmp.shape[1])]) 
# X_train_scaled_new = pd.concat([X_train_scaled, inter_items], axis=1, ignore_index=True)


inter_items.shape, X_train_scaled.shape, X_train_scaled_new.shape
X_train_scaled_new.head(5)


X_train_scaled_new = np.concatenate((X_train_scaled_selected_inter, tmp), axis=1)
# X_train_scaled.append(inter_items, ignore_index=True)
# X_train_scaled.shape

np.isnan(X_train_scaled_new).sum() #.isna().sum()

X_train_scaled = X_train_scaled_new

X_train_scaled_reduced = X_train_scaled[sorted_corr_col[1:6]]
X_test_scaled_reduced = X_test_scaled[sorted_corr_col[1:6]]

clf_reduced = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(X_train_scaled_reduced, y_train.values)
y_pred = clf_reduced.predict(X_test_scaled_reduced)
y_pred_proba = clf_reduced.predict_proba(X_test_scaled_reduced)
acc_score = accuracy_score(y_test, y_pred)

acc_score = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
conf_mat = confusion_matrix(y_test, y_pred)

import collections
counter = collections.Counter(y_test.values)
freq = [counter[i]/len(y_test) for i in [-1, 0, 1]]
print(freq)
# print('true pct in testing data of no-wait', y_test(y_test==0))
print('acc', acc_score)
print('log loss pred', logloss)
print('conf matrix (percentage of all data)\n', np.array_str(conf_mat/len(y_test)*100, precision=4, suppress_small=True) )

X_train_scaled_reduced.shape, y_train.shape, X_test_scaled_reduced.shape, y_test.shape

# Multi-class
# LR default
# clf = LogisticRegression(random_state=0).fit(X_train.values, y_train.values)
# y_pred = clf.predict(X_test)

clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(X_train_scaled, y_train.values)
y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)


acc_score = accuracy_score(y_test, y_pred)
# logloss = log_loss(y_test, y_pred)
# logloss_baseline = log_loss(y_test, np.zeros(len(y_test)))

# print(acc_score)
# confusion_matrix(y_test, y_pred)/len(y_test)


acc_score = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
# logloss_baseline = log_loss(y_test, np.zeros(len(y_test)))
# rocauc = roc_auc_score(y_test, y_pred_proba[:, 1])
conf_mat = confusion_matrix(y_test, y_pred)
# tn, fp, fn, tp = conf_mat.ravel()

import collections
counter = collections.Counter(y_test.values)
freq = [counter[i]/len(y_test) for i in [-1, 0, 1]]
print(freq)
# print('true pct in testing data of no-wait', y_test(y_test==0))
print('acc', acc_score)
print('log loss pred', logloss)
print('conf matrix (percentage of all data)\n', np.array_str(conf_mat/len(y_test)*100, precision=4, suppress_small=True) )
# print("absolute values of tn", tn, "fp", fp, "fn", fn, "tp", tp)

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

logistic = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(X_train_scaled, y_train.values)

clf = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=10, cv=5, verbose=2, n_jobs=-1)

y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)


acc_score = accuracy_score(y_test, y_pred)
# logloss = log_loss(y_test, y_pred)
# logloss_baseline = log_loss(y_test, np.zeros(len(y_test)))

# print(acc_score)
# confusion_matrix(y_test, y_pred)/len(y_test)


acc_score = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
# logloss_baseline = log_loss(y_test, np.zeros(len(y_test)))
# rocauc = roc_auc_score(y_test, y_pred_proba[:, 1])
conf_mat = confusion_matrix(y_test, y_pred)
# tn, fp, fn, tp = conf_mat.ravel()

counter = collections.Counter(y_test.values)
freq = [counter[i]/len(y_test) for i in [-1, 0, 1]]
print(freq)
# print('true pct in testing data of no-wait', y_test(y_test==0))
print('acc', acc_score)
print('log loss pred', logloss)
print('conf matrix (percentage of all data)\n', np.array_str(conf_mat/len(y_test)*100, precision=4, suppress_small=True) )

print('best estimator', best_model.best_params_)

# # Binary
# # LR default
# # clf = LogisticRegression(random_state=0).fit(X_train.values, y_train.values)
# # y_pred = clf.predict(X_test)
# y_train_binary = df.loc[train_index, 'label_wait']
# y_test_binary = df.loc[test_index, 'label_wait']

# clf_binary = LogisticRegression(random_state=0).fit(X_train_scaled, y_train_binary.values)
# y_pred_binary = clf_binary.predict(X_test_scaled)
# y_pred_proba_binary = clf_binary.predict_proba(X_test_scaled)

# eval_model(y_test_binary, y_pred_binary, y_pred_proba_binary)

# # LR random search
# logistic_default = LogisticRegression()

# # Create regularization penalty space
# penalty = ['l1', 'l2']
# # Create regularization hyperparameter distribution using uniform distribution
# C = uniform(loc=0, scale=4)
# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)

# random_search = RandomizedSearchCV(logistic_default, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=1)
# clf_random_search = random_search.fit(X_train_scaled, y_train.values)

# y_pred_random_search = clf_random_search.predict(X_test_scaled)
# y_pred_random_search_proba = clf_random_search.predict_proba(X_test_scaled)

# eval_model(y_test, y_pred_random_search, y_pred_random_search_proba)

clf.coef_.shape
clf.intercept_.shape

# #binary model
# coef = clf.coef_[0]
# coef_hist = coef[:7]
# coef_realtime = coef[7:]
# intercept = clf.intercept_[0]
# print(coef_hist.shape, coef_realtime.shape)
# model_param = dict(zip(cols,coef))
# model_param['intercept'] = intercept

# model_param

#multi-class model
coef = clf.coef_
coef_hist = coef[:, :7]
coef_realtime = coef[:, 7:]
intercept = clf.intercept_
print(coef_hist.shape, coef_realtime.shape, intercept.shape)
model_param = dict(zip(cols,coef.T))
model_param['intercept'] = intercept

model_param

X_test_scaled.shape, clf.coef_.T.shape, clf.intercept_.shape

# softmax, multi_class=multinomial in LR
r = np.matmul(X_test_scaled, clf.coef_.T) + clf.intercept_
# print(r)
r_exp = np.exp(r)
print(r_exp.shape, np.sum(r_exp, axis=1).shape)

prob_softmax = r_exp/r_exp.sum(axis=1)[:,None]
print(prob_softmax)
pred_label = np.argmax(prob_softmax, axis=1)
print(pred_label.shape)
print(np.mean(pred_label==2))

# one-vs-rest. not quite correct...
# sum_rest = r_exp[:, :-1].sum(axis=1)
# print(sum_rest.shape)
# p2 = 1/(1+sum_rest)
# p1 = p2 * np.exp(r[:, 1])
# p0 = p2 * np.exp(r[:, 0])
# print(p2)
# print(p1)
# print(p0)
# prob = np.stack((np.array(p0), np.array(p1), np.array(p2))).T
# pred_label = np.argmax(prob, axis=1)

# print(np.mean(pred_label==2))

df_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns, index=X_train.index)

assert all(df_train_scaled.index == train_index), 'X_train index and train_index do not match'

X_train_store_id = df.loc[train_index]['store_id'].to_frame().join(df_train_scaled)
X_train_store_id_agg = X_train_store_id[ ['store_id'] + feat_hist_agg]

print(X_train_store_id_agg.head(5))

num_stores = X_train_store_id_agg['store_id'].nunique()

df_store_level = X_train_store_id_agg.drop_duplicates(keep='first')
print('num_stores', num_stores)
assert df_store_level.shape[0] == num_stores, 'unmatched after dropping dup'


df_store_level[feat_hist_agg].shape, coef_hist.shape

# #binary model
# hist_agg_value = df_store_level[feat_hist_agg] * coef_hist
# hist_agg_value['intercept'] = intercept
# hist_agg_value = hist_agg_value.sum(axis=1)
# hist_agg_value = hist_agg_value.to_frame()
# hist_agg_value.columns = ['value']
# print(hist_agg_value)

#multi-class model
hist_agg_value = df_store_level[feat_hist_agg].dot(coef_hist.T)
# print(hist_agg_value.head(5))
hist_agg_value += intercept
# print(intercept)
# print(hist_agg_value.head(5))
# hist_agg_value['intercept'] = intercept
# hist_agg_value = hist_agg_value.sum(axis=1)
# hist_agg_value = hist_agg_value.to_frame()
# hist_agg_value.columns = ['value']
print(hist_agg_value)

assert all(df_store_level.index == hist_agg_value.index), 'unmatched index'

df_pre_calculated_value = df_store_level['store_id'].to_frame().join(hist_agg_value)
print(df_pre_calculated_value)

df_pre_calculated_value.mean()

#binary model
# default_value = df_pre_calculated_value['value'].mean()
# df_pre_calculated_value.loc[len(df_pre_calculated_value)]=['no_match', default_value] 
# print(df_pre_calculated_value.tail(4))

#multi-class model
# print(df_pre_calculated_value.tail(4))
df_pre_calculated_value.loc['mean'] = df_pre_calculated_value.mean()
df_pre_calculated_value.loc['mean', 'store_id'] = -1
print(df_pre_calculated_value.tail(4))

import notebooksalamode as mode
mode.export_csv(df_pre_calculated_value)

feat_real_time_min_max = {}
for f in feat_real_time:
  feat_real_time_min_max[f] = [X_train[f].min(), X_train[f].max()]
  print('feature:', f, '\n\tmin:', X_train[f].min(), 'max:',  X_train[f].max())
  
print(feat_real_time_min_max)

### Testing for extracted csv/model coef

valid_X_test = df.loc[test_index][['store_id'] + cols]

valid_X_test['store_id']

# check how many stores in training not in testing
# and how many stores in testing not in training
valid_X_test = df.loc[test_index][['store_id'] + cols]
pre_cal_store = df_pre_calculated_value['store_id'].unique()

tmp = valid_X_test["store_id"].unique()

print(pre_cal_store.shape, tmp.shape)

print('in pre_cal not in testing', len(set(pre_cal_store) - set(tmp)))
print('in testing not in pre_cal', len(set(tmp) - set(pre_cal_store)), 'pct', len(set(tmp) - set(pre_cal_store))/len(pre_cal_store))

valid_X_test_pre_cal = pd.merge(valid_X_test, df_pre_calculated_value, on='store_id', how='left')
assert valid_X_test_pre_cal.shape[0] == valid_X_test.shape[0], 'unmatched shape'
assert valid_X_test_pre_cal['value'].isna().sum() == 0, 'na found'

result = valid_X_test_pre_cal['value']
+10.64552989 * (valid_X_test_pre_cal['tip']/20000.0)
-22.46182226 * (valid_X_test_pre_cal['flf']/19.0)
+46.03914325 * (valid_X_test_pre_cal['pred_horizon']/8575.0)

proba = 1/(1+np.exp(-result))
proba
# binary = proba >=0.5


# X_train_std = np.std(X_train, 0)
X_train_std = np.std(X_train_scaled, 0)

importance_score = [i * j for i, j in zip (list(X_train_std), list(clf.coef_))][0]

res = dict(zip(cols, list(importance_score)))
res = {k: v for k, v in sorted(res.items(), key=lambda item: abs(item[1]), reverse=True)}
print('Feature importance')
print(res)
plt.bar(res.keys(), height=res.values())
plt.xticks(rotation=90)

df_wait = df['wait_before_ready_time']
df_wait[(df_wait < df_wait.quantile(.9999)) & (df_wait > df_wait.quantile(.001))].hist(alpha=0.5)

plt.legend()
plt.title('Histogram of wait_before_ready_time (all data)')

def check_if_no_wait(X, y, y_pred, pred_horizon_thr):
  close_index = X[X['pred_horizon'] < pred_horizon_thr].index
  df_y_pred = pd.DataFrame(data=y_pred, index=y.index)
  pct = df_y_pred.loc[close_index].mean().values[0]
  print(pct, 'are predicted as wait when pred_horizon is', pred_horizon_thr, 'seconds')
  return pct

#check if predict 'no wait' when time is close to food ready time
pct = []
pred_horizon_thr = [240, 300, 360, 480, 600, 720, 840, 960]
for thr in pred_horizon_thr:
  pct.append(check_if_no_wait(X_test, y_test, y_pred, thr))

plt.plot(pred_horizon_thr, pct)
plt.xlabel('pred_horizon (seconds)')
plt.ylabel('pct of wait predicted by model')

test_sampled_index = X_test.sample(1000).index

X_test_sampled = X_test.loc[test_sampled_index].copy()
y_test_sampled = y_test.loc[test_sampled_index].copy()

res = []
ranges = range(0, 60*13, 60)
for var in ranges:
  # print(var)
  X_test_sampled['pred_horizon'] = var
  X_test_sampled_scaled = scaler.transform(X_test_sampled)
  y_pred_sampled_scaled = clf.predict(X_test_sampled_scaled)
  y_pred_sampled_scaled_proba = clf.predict_proba(X_test_sampled_scaled)
  # eval_model(y_test_sampled, y_pred_sampled_scaled, y_pred_sampled_scaled_proba)
  res.append(y_pred_sampled_scaled)

res = np.array(res)
res_mean = res.mean(axis=1)

plt.plot(list(ranges), res_mean)
plt.title('Randomly sampled from testing, and artificially created pred_horizon')
plt.xlabel('pred_horizon (seconds)')

# def check_if_is_late(X, y, y_pred, late_thr):
#   late_index = X[X['wait_before_ready_time'] < -late_thr].index
#   df_y_pred = pd.DataFrame(data=y_pred, index=y.index)
#   pct = df_y_pred.loc[late_index].mean().values[0]
#   print(pct, 'are predicted as wait but is late by', late_thr, 'seconds')
#   return pct

# late_thr = [120, 240, 360, 480, 600, 720, 840, 960]

# pct = []
# for thr in late_thr:
#   pct.append(check_if_is_late(df.loc[test_index], y_test, y_pred, thr))

# # plt.plot(late_thr, pct)
# # plt.xlabel('late (seconds)')
# # plt.ylabel('pct of late but was predicted as wait')

df_test = df.loc[test_index]

df_y_pred = pd.DataFrame(data=y_pred, index=y_test.index, columns=['prediction'])
df_test_wait = df_y_pred[df_y_pred['prediction'] == True]
wait_index = df_test_wait.index

print('all test', df_test.shape, 'test_wait', wait_index.shape)

assert df_test.shape[0] > df_test_wait.shape[0], 'incorrect test and test_wait shape'

def get_late_pct(thr):
  df_test_late = df_test[df_test['wait_before_ready_time'] < -thr]
  late_index = df_test_late.index

  wait_and_late_index = wait_index.intersection(late_index)
  
  print('\t\tlate shape', late_index.shape, 'wait shape', wait_index.shape, 'wait_and_late shape', wait_and_late_index.shape)
  pct = wait_and_late_index.shape[0]/wait_index.shape[0]
  
  return pct

late_thr = [0, 1, 3, 5, 7, 10, 20]
pcts = []  
for thr in late_thr:
  pct = get_late_pct(thr*60)
  print(thr, pct)
  pcts.append(pct)
  

plt.plot(late_thr, pcts)
plt.xlabel('threshold(minutes) to define lateness')
plt.ylabel('pct of pred_wait_but_late / pred_wait ')

df.loc[wait_index]['pred_horizon'].hist()
plt.title('Histogram for all points pred as wait')
plt.xlabel('pred_horizon (seconds)')
plt.ylabel('Freq')
print('min', df.loc[wait_index]['pred_horizon'].min(), 'mean', df.loc[wait_index]['pred_horizon'].mean(), )

flfs = [1.0, 1.5, 2.0]
tips = [300, 400, 500]

pred_horizon_sum = []
for flf in flfs:
  for tip in tips:
    pred_horizon_thr = (-df_pre_calculated_value['value'] 
                        - (tip*feat_coef['tip']-feat_real_time_min_max['tip'][0])/feat_real_time_min_max['tip'][1]\
                        - (flf*feat_coef['flf']-feat_real_time_min_max['flf'][0])/feat_real_time_min_max['flf'][1]) \
                        / ((feat_coef['pred_horizon']-feat_real_time_min_max['pred_horizon'][0])/ feat_real_time_min_max['pred_horizon'][1])

    pred_horizon_thr_clean = pred_horizon_thr[(pred_horizon_thr < pred_horizon_thr.quantile(.99)) & (pred_horizon_thr > pred_horizon_thr.quantile(.01))]
    pred_horizon_thr_clean.hist(alpha=0.5)
    pred_horizon_sum.append(pred_horizon_thr_clean.mean())
    
np.array(pred_horizon_sum).mean()

df_pre_calculated_value['value']

df_tta = datasets['5_analysis_tta']
df_tta.columns = map(str.lower, df_tta.columns)
print(df_tta.head(5))

df_model = df.loc[test_index]
df_model['is_wait'] = df_model['wait_before_ready_time'] > wait_thr
print(df_model.head(5))
# df_comp = df_model.join()

assert df_tta['delivery_id'].nunique() == df_tta.shape[0], 'duplicated delivery_ids in model prediction'
assert df_model['delivery_id'].nunique() == df_model.shape[0], 'duplicated delivery_ids in tta'

assert df_model.shape[0] == y_pred.shape[0], 'shape of x_test and y_test does not match'
assert all(df_model.index == y_test.index), 'index of x_test and y_test does not match'

df_y_test = y_test.to_frame()
df_y_test.columns = ['model_is_wait']
print(df_y_test)

df_model = df_model.join(df_y_test)
print(df_model)

df_comp = pd.merge(df_model[['delivery_id', 'is_wait', 'model_is_wait', 'original_timestamp']], df_tta[['tta_is_assign', 'delivery_id','original_timestamp']], \
                   left_on='delivery_id', right_on='delivery_id', how='inner',\
                   suffixes=('_model', '_tta'))
print(df_comp[['original_timestamp_model', 'original_timestamp_tta']])

df_comp.loc[df_comp['original_timestamp_model'] < df_comp['original_timestamp_tta'], 'tta_is_assign'] = 0

print(df_comp[['tta_is_assign']].mean())

print('pct of model decision making time >= tta assign time', (df_comp['original_timestamp_model'] >= df_comp['original_timestamp_tta']).mean())


num_total = df_comp.shape[0]
print('total num of data points', num_total)
num_stopped_by_model_missed_by_tta = df_comp[(df_comp['is_wait'] == True) & (df_comp['model_is_wait']==True)& (df_comp['tta_is_assign']==1)].shape[0]
print('correctly stopped by model, but not by tta', num_stopped_by_model_missed_by_tta, '\npct of total',num_stopped_by_model_missed_by_tta/num_total)

num_stopped_by_tta_missed_by_model = df_comp[(df_comp['is_wait'] == True) & (df_comp['model_is_wait']==False)& (df_comp['tta_is_assign']==0)].shape[0]
print('correctly stopped by tta, but not by model', num_stopped_by_tta_missed_by_model, '\npct of total',num_stopped_by_tta_missed_by_model/num_total)

1-df_comp['tta_is_assign'].mean(), df_comp['model_is_wait'].mean(), df_comp['is_wait'].mean()



