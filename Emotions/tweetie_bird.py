# My First NLP Project

#& MY GOAL
# My goal is to use a pretrained AI model to determine which emotion is
# expressed in a given tweet.

#& DATA SET BACKGROUND
# "Emotions" by NIDULA ELGIRIYEWITHANA off of Kaggle
# collection of English Twitter messages meticulously annotated with six 
# fundamental emotions: anger, fear, joy, love, sadness, and surprise. 

import keras
import keras_nlp
import pandas as pd
import os; os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Some shortcuts for ease of use
endl = "\n"
div = "\n" + "-"*50
ttl = endl*2

# Use mixed precision to speed up all training in this guide.
keras.mixed_precision.set_global_policy("mixed_float16")

print(ttl, "start")

#? FIRST STEP: What is in data?
print(ttl, "UNDERSTAND DATASET", div)
raw_df = pd.read_csv('Emotions_Kaggle.csv') 
print("loaded dataset")
print('\nFeatures include:', raw_df.columns, endl)
# Index(['id', 'text', 'label'], dtype='object')
print('Data Types:'); print(raw_df.dtypes, endl)

#* How many data points are there?
print('The number of rows:', raw_df.shape[0], endl)
# 416809 training examples

#* Is there any null/nan values?
print('Is there any null values? -> No')
print(raw_df.isnull().sum(), endl)
# No null values

#* What is the count of each emotion? Is it mostly even?
# sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 
                4: 'fear', 5: 'surprise'}
raw_df['emotion'] = raw_df['label'].map(emotion_map)

print("Count of each emotion:")
print(raw_df['emotion'].value_counts(), endl)
# Mostly joy and sadness, least present one is surprise

#* Remove ID since we don't need that
raw_df.drop(columns=['id'], inplace=True)

#? SECOND STEP: Convert this into smth we can use
print(ttl, "MAKING MODEL", div)

# Split the DataFrame into training and testing sets
raw_df_sample = raw_df.sample(frac=0.01, random_state=42)  # 40% of the data
print("Number of rows:", raw_df.shape[0])
print("Number of rows in subset:", raw_df_sample.shape[0])

train_df, test_df = train_test_split(raw_df_sample, test_size=0.2, random_state=42)  # Adjust test_size as needed

# Convert the DataFrame subsets into TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_df['text'].values, 
                                                train_df['label'].values))
test_ds = tf.data.Dataset.from_tensor_slices((test_df['text'].values, 
                                                test_df['label'].values))

print(train_ds.element_spec, endl)  # Check the structure of the dataset elements

# Shuffle and batch the datasets
BATCH_SIZE = 8
train_ds = train_ds.shuffle(len(train_df)).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

# Optionally, you can also prefetch and cache the datasets for better performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print(train_ds.unbatch().take(1).get_single_element())
print(endl)

classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=6 
)

#* How well does it work for with zero training whatsoever?
classifier.evaluate(test_ds)

trythis = input("Try string: ")
while trythis.lower() != 'done':
    prediction = classifier.predict([trythis])
    print('Classifier prediction:', prediction)
    trythis = input("Try string: ")


# #* Let's train it
# classifier.fit(
#     train_ds,
#     validation_data=test_ds,
#     epochs=1,
# )

print(ttl, "fin")