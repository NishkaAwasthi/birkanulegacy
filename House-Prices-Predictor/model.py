import pandas as pd # to load the dataframe
import matplotlib.pyplot as plt # to visualize the data
import seaborn as sns # To see the correlation between features using heatmap
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Some shortcuts for ease of use
endl = "\n"
div = "\n" + "-"*87
ttl = endl*2

# Load datasetc
raw_dataset = pd.read_excel("HousePricePrediction.xlsx") # needed openpyxl

#? What is in the RAW DATA?
print(ttl, "1. RAW DATA INFORMATION:", div)
print(raw_dataset.head(5)) # see first 5 training examples

print("\nSHAPE: ", raw_dataset.shape)
#& there are 2919 training + testing examples and 13 features.

print("\nFEATURES: ", end="")
for i in raw_dataset.columns:
    print(i, end = ", ")
    if i == 'YearBuilt': print()
print()
# 'Id', 'MSSubClass', 'MSZoning', 'LotArea', 'LotConfig', 'BldgType',
#  'OverallCond', 'YearBuilt', 'YearRemodAdd', 'Exterior1st', 
#  'BsmtFinSF2', 'TotalBsmtSF', 'SalePrice'

#? Data Preprocessing- What types of data are there?
print(ttl, "2. DATA PREPROCESSING:", div)
print(raw_dataset.dtypes, endl)
#& int64, object, and float64 datatypes present in dataset 

print("DATA TYPES: ")
obj = (raw_dataset.dtypes == 'object')
cat_cols = list(obj[obj].index)
print("Categorical variables:",len(cat_cols))
 
int_ = (raw_dataset.dtypes == 'int64')
print("Integer variables:",len(list(int_[int_].index)))
 
fl = (raw_dataset.dtypes == 'float64')
print("Float variables:",len(list(fl[fl].index)))

num_cols = list(int_[int_].index) + list(fl[fl].index)

#& 4 Categorical, 6 Integer, 3 Float 


#? Data Exploratory Analysis- What is in each feature?
print(ttl, "3. DATA EXPLORATION:", div)

# Generate heatmap based on correlation matrix - NUMERICAL FEATURES ONLY
num_cols = raw_dataset.select_dtypes(include=['float64', 'int64'])
print("Correlation Table:")
print(num_cols.corr())

plt.figure(figsize=(12, 12))
htmp = sns.heatmap(num_cols.corr(), 
                        annot=True, 
                        cmap='pink', 
                        fmt = '.2f',
                        linewidths = 2)
htmp.figure.savefig('heatmap-numerical-features.png')
print(endl, "Heatmap generated to 'heatmap-numerical-features.png'.", endl)
plt.clf()

#& Correlations seem to make sense logically, 
#& Basement SF has highest correlation with Sales Price 
#& Had to exclude categorical variables

# Generate bar chart - CATEGORICAL FEATURES
uni_vals = {}
uv_cnt = []
for col in cat_cols:
    uv_cnt.append(len(list(raw_dataset[col].unique())))
    uni_vals[col] =  [len(list(raw_dataset[col].unique())), list(raw_dataset[col].unique())]

print("Categorical Features:")
for i in range(len(cat_cols)):
    print(cat_cols[i], f"({uni_vals[cat_cols[i]][0]}):", uni_vals[cat_cols[i]][1])

plt.figure(figsize=(12, 12))
plt.title('Number of Unique values of each Categorical Feature')
plt.xticks(rotation=90)
bplt = sns.barplot(x=cat_cols, y=uv_cnt, 
                    hue=cat_cols, 
                    palette='flare', 
                    legend=False)
bplt.figure.savefig('barplot-categorical-features.png')
print(endl, "Barplot generated to 'barplot-categorical-features.png'.")
plt.clf()

#& The 4 categorical variables that have 6, 5, 5, and 16 unique 
#& values, respectively. 
#! This will not work for template!
plt.figure(figsize=(18, 36))
plt.suptitle('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1
num_cols = 4 
num_rows = (len(cat_cols) + num_cols - 1) // num_cols  
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 36), sharey=True)
plt.subplots_adjust(hspace=0.5, wspace=0.4)

for col, ax in zip(cat_cols, axes.flatten()):
    y = raw_dataset[col].value_counts()
    sns.barplot(x=list(y.index), y=y, 
                ax=ax, palette='flare', hue=list(y.index))
    ax.set_xticks(range(len(y.index)))  # Set tick positions
    ax.set_xticklabels(y.index, rotation=90)  # Set tick labels
    ax.set_title(col)

plt.tight_layout()  # Adjust layout to prevent overlapping titles
plt.subplots_adjust(top=0.95, bottom=0.05)  # Adjust top and bottom margins
plt.savefig('categorical_distribution.png')

#? Data Cleaning- Remove/Substitute unneeded data
print(ttl, "4. DATA CLEANING:", div)
#! Cannot be template code, needs to be personalized for each data set

#& Since we do not need ID, we can drop it.
clean_data = raw_dataset.copy()
clean_data.drop(['Id'],
                    axis=1, # columns
                    inplace=True) # do on existing df

#& Check for any null or missing values.
null_cnt = clean_data.isnull().sum()

# Print the columns with null values and their respective counts
print("Columns with null values:")
for col, count in null_cnt.items():
    if count > 0:
        print(f"{col}: {count}")

#& Obviously, there is a significant number of missing data for sales price
#& We can fill it in with the mean.
clean_data['SalePrice'] = clean_data['SalePrice'].fillna(
  clean_data['SalePrice'].mean())

#& Since there are only a handful of null rows left, we can just drop them
clean_data = clean_data.dropna()
 
print('Data cleaning completed.')
print(endl, "CLEAN DATA SHAPE: ", clean_data.shape)

# Cleaned data in final data for easier logical use
final_data = clean_data.copy()
#& We now have 2913 training examples and 12 features

#? Data Preparation- 
print(ttl, "5. DATA PREPARATION:", div)
#& Categorical variable transformation: We are using one hot encoding, 
#& that will create a binary variable for each category for each feature

s = (final_data.dtypes == 'object')
cat_cols_clnd = list(s[s].index)
print("Categorical variables:")
print(cat_cols)

# Encode categorical features as a one-hot numeric array.
OH_encoder = OneHotEncoder(sparse_output=False) 

OH_cols = pd.DataFrame(OH_encoder.fit_transform(final_data[cat_cols_clnd]))

OH_cols.index = final_data.index
OH_cols.columns = OH_encoder.get_feature_names_out()
final_data = final_data.drop(cat_cols_clnd, axis=1)
final_data = pd.concat([final_data, OH_cols], axis=1)

print("Categorical features transformation completed.")

#& Separate these things into 2 different data sets- training and valid
#& X and Y splitting: linear regression means that h(x) -> y
X = final_data.drop(['SalePrice'], axis=1)
Y = final_data['SalePrice']


# Split the training set into training and validation set
# train_test_split does this for us randomly
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=1) 
    # 80% training, 20% validation

print("Data splitting completed. 80% training, 20% validation")

#? Model Building
print(ttl, "6. MODEL BUILDING:", div)
#& Building linear regression model using sci-kit
linreg_model = LinearRegression()
linreg_model.fit(X_train, Y_train)
Y_pred = linreg_model.predict(X_valid)

#& Y_pred and Y_valid comparison
r_sq = r2_score(Y_valid, Y_pred)
mape = mean_absolute_percentage_error(Y_valid, Y_pred)
print("R-squared score:", r_sq)
print("MAPE:", mape)

#& We would want an R_sq score closer to 1 & a lower mape score

#? Multiple Model Building
print(ttl, "7. MANIPULATING THE NUMBER OF FEATURES:", div)
#& Model only with top 3 based on heat map
print("What if we had a model of only the top 3 correlations?: ")
X = final_data.loc[:, ['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF']]
Y = final_data['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=1) 
    # 80% training, 20% validation

top3_model = LinearRegression()
top3_model.fit(X_train, Y_train)
Y_pred = top3_model.predict(X_valid)

r_sq_top3 = r2_score(Y_valid, Y_pred)
mape_top3 = mean_absolute_percentage_error(Y_valid, Y_pred)
print("R-squared score:", r_sq_top3)
print("MAPE:", mape_top3)

print()
if r_sq_top3 > r_sq: print("Top3 Model worked better for r_sq")
else: print("Full Model worked better for r_sq")

if mape_top3 < mape: print("Top3 Model worked better for mape")
else: print("Full Model worked better for mape")

#& Model only with categorical variables
print(endl, "What if we had a model of only the categorical features?: ")

arr = ['SalePrice', 'MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF']
X = final_data.drop(arr, axis=1)
Y = final_data['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=1) 
    # 80% training, 20% validation

cat_model = LinearRegression()
cat_model.fit(X_train, Y_train)
Y_pred = cat_model.predict(X_valid)

r_sq_cat = r2_score(Y_valid, Y_pred)
mape_cat = mean_absolute_percentage_error(Y_valid, Y_pred)
print("R-squared score:", r_sq_cat)
print("MAPE:", mape_cat)

print()
if r_sq_cat > r_sq: print("Categorical Features Model worked better for r_sq")
else: print("Full Model worked better for r_sq")

if mape_cat < mape: print("Categorical Features Model worked better for mape")
else: print("Full Model worked better for mape")

#& Model only with numerical variables
print(endl, "What if we had a model of only the numerical features?: ")
X = clean_data.drop(['SalePrice', 'MSZoning', 'LotConfig', 'BldgType', 'Exterior1st'], axis=1)
Y = clean_data['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=1) 
    # 80% training, 20% validation

num_model = LinearRegression()
num_model.fit(X_train, Y_train)
Y_pred = num_model.predict(X_valid)

r_sq_num = r2_score(Y_valid, Y_pred)
mape_num = mean_absolute_percentage_error(Y_valid, Y_pred)
print("R-squared score:", r_sq_num)
print("MAPE:", mape_num)

print()
if r_sq_num > r_sq: print("Numerical Features Model worked better for r_sq")
else: print("Full Model worked better for r_sq")

if mape_num < mape: print("Numerical Features Model worked better for mape")
else: print("Full Model worked better for mape")


print(endl, "DONE", endl*2)
