#!/usr/bin/env python
# coding: utf-8

# # Project:- Gender and Age Detection Python Project
# Organization:- Infopillar Solution

# # Data Science Internship
# Author:- Arshad R. Bagde

# In[ ]:





# In[45]:


get_ipython().system('pip install opencv')


# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#load data
fold0 = pd.read_excel("fold_0_data.xlsx" )
fold1 = pd.read_excel("fold_1_data.xlsx")
fold2 = pd.read_excel("fold_2_data.xlsx")
fold3 = pd.read_excel("fold_3_data.xlsx")
fold4 = pd.read_excel("fold_4_data.xlsx")


# In[3]:


df = pd.concat([fold0, fold1, fold2, fold3, fold4], ignore_index=True, axis = 0).reset_index(drop = True)
print(fold0.shape, fold1.shape, fold2.shape, fold3.shape, fold4.shape, df.shape)


# In[4]:


print(df.shape)
df.head()


# In[5]:


df.dropna(how = 'any').shape


# In[6]:


df.duplicated().sum()


# In[7]:


df.drop_duplicates(keep = 'first').shape


# In[8]:


df.drop_duplicates(keep = False).shape


# In[9]:


df.dropna(subset = ['user_id', 'original_image', 'face_id', 'age', 'gender', 'x', 'y', 'dx', 'dy', 'tilt_ang', 'fiducial_yaw_angle', 
                    'fiducial_score'], how = 'any').shape


# In[10]:


df.info()


# In[11]:


df.columns


# In[12]:


df.columns.to_series().groupby(df.dtypes).groups


# In[13]:


df.isnull().sum()


# In[14]:



df.nunique()


# In[15]:


df.notnull().tail()


# In[16]:


df.apply(lambda x: x.dtype)


# In[17]:


round((df.apply(lambda x:x.isnull().sum())/len(df))*100,2)


# In[18]:


#Checking for percentage of missing values in each columns
(df.isnull().sum()/len(df))*100


# In[19]:


total_miss = df.isnull().sum()
perc_miss = total_miss/df.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,'% missing':perc_miss})

missing_data.sort_values(by='Total missing',ascending=False).head(3)


# In[20]:


print('Unique Values for Each Feature: \n')
for i in df.columns:
    print(i, ':',df[i].nunique()) 


# In[21]:


# find the unique values from categorical features
for col in df.select_dtypes(include='object').columns:
    print(col)
    print(df[col].unique())


# In[22]:


df.describe()


# In[23]:


df.corr()


# In[24]:


df.describe(include = object)


# In[25]:



userid_Series = df['user_id']
userid_levels = userid_Series.unique()
userid_levels


# In[26]:


originalimage_Series = df['original_image']
originalimage_levels = originalimage_Series.unique()
originalimage_levels


# In[27]:


age_Series = df['age']
age_levels = age_Series.unique()
age_levels


# In[28]:


gender_Series = df['gender']
gender_levels = gender_Series.unique()
gender_levels


# In[29]:


userid_freq_table = pd.crosstab(index = df['user_id'], columns = 'freq')
userid_freq_table


# In[30]:


age_freq_table = pd.crosstab(index = df['age'], columns = 'freq')
age_freq_table


# In[31]:


gender_freq_table = pd.crosstab(index = df['gender'], columns = 'freq')
gender_freq_table


# In[32]:


# list of numerical variables
numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['y']))]
print('Number of numerical variables: ', len(numerical_features))


# In[33]:


#Discrete Numerical Features
discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[34]:



#Continuous Numerical Features
continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['deposit']]
print("Continuous feature Count: {}".format(len(continuous_features)))


# In[35]:


df.hist(figsize=(20,20))
plt.show()


# In[36]:


matrix = df.corr() 
f, ax = plt.subplots(figsize=(25, 12)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="RdYlGn",annot = True);


# In[37]:


df = df.dropna()
#print(df)
df.head()


# In[38]:



df.dropna(inplace = True)
df.isnull().sum()


# In[39]:


### categorical
categorical_cols = list(df.select_dtypes(include=['object']))
categorical_cols


# In[40]:


def Count_categorcial_variables(df):
    categorcial_variables = df.select_dtypes(include=['object']).columns.tolist()
    #fig = plt.figure(figsize=(14, 18))

    for index, col in enumerate(categorcial_variables):
        print("------------",col," value counts---------------------")
        print(df[col].value_counts())
        #fig.add_subplot(3, 2, index+1)
        #dataframe[col].value_counts()[:20].plot(kind='bar', title=col, color = "royalblue")
        #plt.tight_layout()
        
    print("\n\n------------Number of categories in each columns---------------------")
    for i in categorcial_variables:
        a = df[i].unique()
        print("There are {} categories in {}".format(len(a),i))
Count_categorcial_variables(df)


# In[41]:


df['gender'].value_counts()


# In[42]:


plt.figure(figsize=(10,10))
sns.countplot(y = df['gender'])


# In[43]:


plt.figure(figsize=(10,10))
df['gender'].value_counts().plot.pie(autopct="%0.2f%%")


# In[44]:


df['age'].value_counts()


# In[45]:


plt.figure(figsize=(10,10))
sns.countplot(y = df['age'])


# In[46]:



plt.figure(figsize=(10,10))
df['age'].value_counts().plot.pie(autopct="%0.2f%%")


# In[47]:


#bar chart
gender = ['f','m','u']
plt.bar(gender, df.gender.value_counts(), align='center', alpha=0.5)
plt.show()


# In[48]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Dropout, LayerNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# In[49]:



path = "E:/Internshala/2-InfoPillar/Task/Task-1-Gender and Age Detection Python Project/AdienceBenchmarkGenderAndAgeClassification/AdienceBenchmarkGenderAndAgeClassification/faces/10354155@N05/coarse_tilt_aligned_face.424.11937896894_147ae7c049_o.jpg"
img = load_img(path)
plt.imshow(img)
plt.show()


# In[50]:


imp_data = df[['age', 'gender', 'x', 'y', 'dx', 'dy']].copy()
imp_data.info()
img_path = []
for row in df.iterrows():
    path = "E:/Internshala/2-InfoPillar/Task/Task-1-Gender and Age Detection Python Project/AdienceBenchmarkGenderAndAgeClassification/AdienceBenchmarkGenderAndAgeClassification/faces/10354155@N05/coarse_tilt_aligned_face.424.11937896894_147ae7c049_o.jpg"
    img_path.append(path)
imp_data['img_path'] = img_path
imp_data.head()


# In[51]:


imp_data = imp_data.dropna()
clean_data = imp_data[imp_data.gender != 'u'].copy()
clean_data.info()


# In[52]:


gender_to_label_map = {
    'f' : 0,
    'm' : 1
}
clean_data['gender'] = clean_data['gender'].apply(lambda g: gender_to_label_map[g])
clean_data.head()


# In[53]:


clean_data['age'] = clean_data['age'].map({'(25, 32)': 0, '(0, 2)': 1, '(38, 43)': 2, '(4, 6)': 3, '(8, 12)': 4, '(15, 20)': 5, 
                                           '(60, 100)': 6, '(48, 53)': 7, '35': 8, '13': 9, '22': 10, '34': 11, '23': 12, '45': 13, 
                                           '(27, 32)': 14, '55': 15, '36': 16, '(38, 42)': 17, 'None': 18, '57': 19, '3': 20, '29': 21, 
                                           '(38, 48)': 22, '58': 23, '2': 24, '42': 25, '(8, 23)': 26, '46': 27})
clean_data.head()


# In[54]:



X = clean_data[['img_path']]
y = clean_data[['gender']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Train data shape {}'.format(X_train.shape))
print('Test data shape {}'.format(X_test.shape))
train_images = []
test_images = []
for row in X_train.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((227, 227))   # Resize the image
    data = np.asarray(image)
    train_images.append(data)
for row in X_test.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((227, 227))  # Resize the image
    data = np.asarray(image)
    test_images.append(data)
train_images = np.asarray(train_images)
test_images = np.asarray(test_images)
print('Train images shape {}'.format(train_images.shape))
print('Test images shape {}'.format(test_images.shape))


# In[55]:



model = Sequential()
model.add(Conv2D(input_shape=(227, 227, 3), filters=96, kernel_size=(7, 7), strides=4, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(LayerNormalization())
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(LayerNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(LayerNormalization())
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=2, activation='softmax'))
model.summary()


# In[56]:


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) # Callback for earlystopping
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, y_train, batch_size=32, epochs=25, validation_data=(test_images, y_test), callbacks=[callback])
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
model.save('gender_model25.h5')


# In[57]:


test_loss, test_acc = model.evaluate(test_images, y_test, verbose=2)
print(test_acc)


# In[58]:



X = clean_data[['img_path']]
y = clean_data[['age']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Train data shape {}'.format(X_train.shape))
print('Test data shape {}'.format(X_test.shape))
train_images = []
test_images = []
for row in X_train.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((227, 227))   # Resize the image
    data = np.asarray(image)
    train_images.append(data)
for row in X_test.iterrows():
    image = Image.open(row[1].img_path)
    image = image.resize((227, 227))  # Resize the image
    data = np.asarray(image)
    test_images.append(data)
train_images = np.asarray(train_images)
test_images = np.asarray(test_images)
print('Train images shape {}'.format(train_images.shape))
print('Test images shape {}'.format(test_images.shape))


# In[59]:



model = Sequential()
model.add(Conv2D(input_shape=(227, 227, 3), filters=96, kernel_size=(7, 7), strides=4, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(LayerNormalization())
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(LayerNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(LayerNormalization())
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=8, activation='softmax'))
model.summary()


# In[ ]:





# In[ ]:


***Thank You***


# In[ ]:


***The End***


# In[ ]:


#Reference:- https://www.kaggle.com/depekha/gender-and-age-detection

