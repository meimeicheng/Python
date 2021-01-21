import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from tensorflow.keras import models, layers
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.backend import get_graph
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime



def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

def preprocessing(df):

    dfresult = pd.DataFrame()
    
    # Let's keep features in the original order

    # Pclass -> one-hot encoding
    pclass = pd.get_dummies(df['Pclass'])
    pclass.columns = ['Pclass_' + str(x) for x in pclass.columns ]
    dfresult = pd.concat([dfresult, pclass],axis = 1)

    # Sex -> one-hot encoding
    sex = pd.get_dummies(df['Sex'])
    dfresult = pd.concat([dfresult,sex],axis = 1)

    # Age -> replace with imputation
    dfresult['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
    # SibSp, Parch, Fare
    dfresult['SibSp'] = df['SibSp']
    dfresult['Parch'] = df['Parch']
    dfresult['Fare'] = df['Fare']

    # Embarked -> one-hot encoding
    embarked = pd.get_dummies(df['Embarked'],dummy_na=True)
    embarked.columns = ['Embarked_' + str(x) for x in embarked.columns]
    dfresult = pd.concat([dfresult, embarked],axis = 1)

    return dfresult

def train_model(x_test, y_test):
    tf.keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(10,activation = 'relu',input_shape=(13,)))
    model.add(layers.Dense(20,activation = 'relu' ))
    model.add(layers.Dense(1,activation = 'sigmoid' ))
    model.compile(optimizer='sgd',
            loss='binary_crossentropy',
            metrics=['accuracy'])

    logdir = "logs2/hparam_tuning"
    writer = tf.summary.create_file_writer(logdir)
    # Forward pass
    with writer.as_default():
      if not model.run_eagerly:
        summary_ops_v2.graph(get_graph(), step=0)

    history = model.fit(x_train,y_train,
                    batch_size= 64,
                    epochs= 30,
                    validation_split=0.2
                   )
    return model


df_train_raw = pd.read_csv('input2/train.csv')
df_test_raw = pd.read_csv('input2/test.csv')

# train_copy = df_train_raw.copy() # For test only
# train_copy['Age'] = train_copy[['Age','Pclass']].apply(impute_age, axis=1)

x_train = preprocessing(df_train_raw)
print(x_train)
y_train = df_train_raw['Survived'].values

x_test = preprocessing(df_test_raw)
y_test = df_test_raw['Survived'].values

print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )
# Convert dataframe data into np array
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

model = train_model(x_test,y_test)
model.summary()
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
model.evaluate(x = x_test,y = y_test)

#hparam tunning
HP_NUM_UNITS_ONE = hp.HParam('num_units_one', hp.Discrete([5, 10, 20]))
HP_NUM_UNITS_TWO = hp.HParam('num_units_two', hp.Discrete([10, 20, 40]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs2/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS_ONE, HP_NUM_UNITS_TWO, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams):
  model = tf.keras.models.Sequential()
  model.add(layers.Dense(hparams[HP_NUM_UNITS_ONE], activation = 'relu',input_shape=(13,)))
  model.add(layers.Dense(hparams[HP_NUM_UNITS_TWO],activation = 'relu' ))
  model.add(layers.Dense(1,activation = 'sigmoid' ))
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='binary_crossentropy',
      metrics=['accuracy'],
  )


  model.fit(x_train, y_train, epochs=30) 
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0
for num_units_one in HP_NUM_UNITS_ONE.domain.values:
  for num_units_two in HP_NUM_UNITS_TWO.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS_ONE: num_units_one,
          HP_NUM_UNITS_TWO: num_units_two,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('>> Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs2/hparam_tuning/' + run_name, hparams)
      session_num += 1