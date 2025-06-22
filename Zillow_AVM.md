# Zillow Home Value Prediction 

<p style="margin-top:50px;"></p>

### Kaggle project link: https://www.kaggle.com/competitions/zillow-prize-1

“Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. And, by continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning.

The **objective** of this project is to predict the log error which is defined as:

$$
\text{logerror} = \log(\text{Zestimate}) - \log(\text{SalePrice})
$$

and it is recorded in the transactions training data provided. 

If a transaction didn't happen for a property during that period of time, that row is ignored and not counted.

For each property (unique parcelid), you must predict a log error for each time point. You should be predicting 6 timepoints: October 2016 (201610), November 2016 (201611), December 2016 (201612), October 2017 (201710), November 2017 (201711), and December 2017 (201712).

# Data Preperation


```python
import os
os.chdir('C:\\Users\\bayle\\Directory')
```


```python
import pandas as pd
train_2016 = pd.read_csv('train_2016_v2.csv') #target variable (and  time stamp) 
train_2017 = pd.read_csv('train_2017.csv') 
properties_2016 = pd.read_csv('properties_2016.csv', low_memory=False) #property features 
properties_2017 = pd.read_csv('properties_2017.csv', low_memory=False) #identical features, different year
```


```python
train_2016.shape
```




    (90275, 3)




```python
train_2017.shape
```




    (77613, 3)



**train_2017 only contains data from 01-2017 to 09-2017**


```python
train_2017.transactiondate
```




    0        2017-01-01
    1        2017-01-01
    2        2017-01-01
    3        2017-01-01
    4        2017-01-01
                ...    
    77608    2017-09-20
    77609    2017-09-20
    77610    2017-09-21
    77611    2017-09-21
    77612    2017-09-25
    Name: transactiondate, Length: 77613, dtype: object




```python
properties_2016.shape
```




    (2985217, 58)




```python
properties_2017.shape
```




    (2985217, 58)



 

### Let's have a look at the proportion of Null values in properties_2016


```python
properties_2016.isnull().mean().sort_values(ascending=False) * 100
```




    storytypeid                     99.945599
    basementsqft                    99.945465
    yardbuildingsqft26              99.911330
    fireplaceflag                   99.827048
    architecturalstyletypeid        99.796966
    typeconstructiontypeid          99.773986
    finishedsquarefeet13            99.743000
    buildingclasstypeid             99.576949
    decktypeid                      99.427311
    finishedsquarefeet6             99.263002
    poolsizesum                     99.063385
    pooltypeid2                     98.925539
    pooltypeid10                    98.762603
    taxdelinquencyflag              98.108613
    taxdelinquencyyear              98.108546
    hashottuborspa                  97.688141
    yardbuildingsqft17              97.308236
    finishedsquarefeet15            93.608572
    finishedfloor1squarefeet        93.209304
    finishedsquarefeet50            93.209304
    threequarterbathnbr             89.560859
    fireplacecnt                    89.527160
    pooltypeid7                     83.737899
    poolcnt                         82.663438
    numberofstories                 77.151778
    airconditioningtypeid           72.815410
    garagetotalsqft                 70.411967
    garagecarcnt                    70.411967
    regionidneighborhood            61.262381
    heatingorsystemtypeid           39.488453
    buildingqualitytypeid           35.063749
    unitcnt                         33.757244
    propertyzoningdesc              33.719090
    lotsizesquarefeet                9.248875
    finishedsquarefeet12             9.246664
    calculatedbathnbr                4.318346
    fullbathcnt                      4.318346
    censustractandblock              2.516601
    landtaxvaluedollarcnt            2.268947
    regionidcity                     2.105207
    yearbuilt                        2.007492
    calculatedfinishedsquarefeet     1.861339
    structuretaxvaluedollarcnt       1.841809
    taxvaluedollarcnt                1.425357
    taxamount                        1.046825
    regionidzip                      0.468308
    propertycountylandusecode        0.411260
    roomcnt                          0.384394
    bathroomcnt                      0.383959
    bedroomcnt                       0.383557
    assessmentyear                   0.383188
    longitude                        0.383121
    fips                             0.383121
    latitude                         0.383121
    propertylandusetypeid            0.383121
    rawcensustractandblock           0.383121
    regionidcounty                   0.383121
    parcelid                         0.000000
    dtype: float64



From this inital analysis I have concluded that there are many features present which lack completeness.

This is driven by unique features that only apply to a small number of properties (such as, pools type, basements, yard buildings, fireplaces, garage size etc.) Including some of these features would create sparse vectors which increase the risk of overfitting.

For detail see: Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection. Journal of Machine Learning Research, 3(Mar):1157–1182. Link: https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf

**We will drop features which have > 5% incompleteness.** 


```python
def drop_features(df, threshold):
    """
    Drops columns with missing values above a certain threshold.
    
    :param df: DataFrame to process
    :param threshold: Proportion of missing values to decide column removal, e.g. 0.95 removes columns with more than 95% of values missing
    :return: Cleaned DataFrame
    """
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    
    return df.drop(columns=cols_to_drop)

properties_2016 = drop_features(properties_2016, 0.95) #drop features of > 5% incompleteness
properties_2017 = drop_features(properties_2017, 0.95)
```


```python
properties_2016.shape
```




    (2985217, 41)



Before joining our data, we will verify the following:

1. parcelid exists in both the train and properties datasets and is unique. 
2. All unique parcelid's in the train datasets are present in the properties datasets.
3. Features in the 2016 and 2017 datasets are the same.


```python
print(f"train_2016 duplicates: {len(train_2016[train_2016['parcelid'].duplicated(keep=False)])}")
print(f"train_2017 duplicates: {len(train_2017[train_2017['parcelid'].duplicated(keep=False)])}")
print(f"properties_2016 duplicates: {len(properties_2016[properties_2016['parcelid'].duplicated(keep=False)])}")
print(f"properties_2017 duplicates: {len(properties_2017[properties_2017['parcelid'].duplicated(keep=False)])}")
```

    train_2016 duplicates: 249
    train_2017 duplicates: 395
    properties_2016 duplicates: 0
    properties_2017 duplicates: 0
    


```python
parcelid_2016 = set(properties_2016['parcelid'])
parcelid_2017 = set(properties_2017['parcelid'])

# In 2016 but not in 2017
only_2016 = parcelid_2016 - parcelid_2017
count_only_2016 = len(only_2016)

# In 2017 but not in 2016
only_2017 = parcelid_2017 - parcelid_2016
count_only_2017 = len(only_2017)

print("ParcelIDs in 2016 but not in 2017:", count_only_2016)
print("ParcelIDs in 2017 but not in 2016:", count_only_2017)
```

    ParcelIDs in 2016 but not in 2017: 0
    ParcelIDs in 2017 but not in 2016: 0
    


```python
#All parcelids in train_2016 can be found in properties_2016
print(train_2016['parcelid'].isin(properties_2016['parcelid']).all())
```

    True
    


```python
if list(train_2016.columns) == list(train_2017.columns):
    print('Columns in train_2016 and train_2017 are identical')
else:
    print('Columns in train_2016 and train_2017 are NOT identical')

if list(properties_2016.columns) == list(properties_2017.columns):
    print('Columns in properties_2016 and properties_2017 are identical')
else:
    print('Columns in properties_2016 and properties_2017 are NOT identical')
```

    Columns in train_2016 and train_2017 are identical
    Columns in properties_2016 and properties_2017 are identical
    

 

**We can see that the parcelid is not unique in the train datasets.**

**There seem to be duplicates in train which indicate a property being transacted multiple times.**

**Duplicates seem to represent instances in which a property is transacted multiple times in a year (Should be clarified with an SME).**


```python
train_2016[train_2016['parcelid'].duplicated(keep=False)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parcelid</th>
      <th>logerror</th>
      <th>transactiondate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>496</th>
      <td>13850164</td>
      <td>-0.1567</td>
      <td>2016-01-05</td>
    </tr>
    <tr>
      <th>497</th>
      <td>13850164</td>
      <td>-0.0460</td>
      <td>2016-06-29</td>
    </tr>
    <tr>
      <th>781</th>
      <td>14677191</td>
      <td>-0.3682</td>
      <td>2016-01-06</td>
    </tr>
    <tr>
      <th>782</th>
      <td>14677191</td>
      <td>-0.0845</td>
      <td>2016-09-12</td>
    </tr>
    <tr>
      <th>813</th>
      <td>11005771</td>
      <td>-0.0131</td>
      <td>2016-01-06</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>74547</th>
      <td>11419032</td>
      <td>-0.0040</td>
      <td>2016-12-29</td>
    </tr>
    <tr>
      <th>78229</th>
      <td>17128287</td>
      <td>-0.0030</td>
      <td>2016-09-22</td>
    </tr>
    <tr>
      <th>78230</th>
      <td>17128287</td>
      <td>0.0090</td>
      <td>2016-11-04</td>
    </tr>
    <tr>
      <th>80673</th>
      <td>14367791</td>
      <td>2.0550</td>
      <td>2016-09-29</td>
    </tr>
    <tr>
      <th>80674</th>
      <td>14367791</td>
      <td>2.1240</td>
      <td>2016-09-30</td>
    </tr>
  </tbody>
</table>
<p>249 rows × 3 columns</p>
</div>




```python
def compare_properties_datasets(df1, df2, pid):
    """
    Allows us to compare the values in each dataset associated with a given property id side-by-side. If it repeats we will just compare the first.
    
    :param df1, df2: DataFrames to process
    :param pid: Unique property id to visualise 
    :return: df_compare, df with two columns one contains the values for df1 and df2
    """

    row1 = df1.loc[df1['parcelid'] == pid]
    row2 = df2.loc[df2['parcelid'] == pid]
    
    if df1.shape[0] == 0 or df2.shape[0] == 0:
        raise ValueError(f"No row found for parcelid {pid} in one of the DataFrames")

    df_compare = pd.DataFrame({
        'properties_2016': df1.iloc[0],
        'properties_2017': df2.iloc[0]
    })
    
    return df_compare

compare_properties_datasets(properties_2016, properties_2017, 10754147)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>properties_2016</th>
      <th>properties_2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>parcelid</th>
      <td>10754147</td>
      <td>10754147</td>
    </tr>
    <tr>
      <th>airconditioningtypeid</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bathroomcnt</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>bedroomcnt</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>buildingqualitytypeid</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calculatedbathnbr</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedfloor1squarefeet</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calculatedfinishedsquarefeet</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedsquarefeet12</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedsquarefeet15</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedsquarefeet50</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fips</th>
      <td>6037.0</td>
      <td>6037.0</td>
    </tr>
    <tr>
      <th>fireplacecnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fullbathcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>garagecarcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>garagetotalsqft</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>heatingorsystemtypeid</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>34144442.0</td>
      <td>34144442.0</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-118654084.0</td>
      <td>-118654084.0</td>
    </tr>
    <tr>
      <th>lotsizesquarefeet</th>
      <td>85768.0</td>
      <td>85768.0</td>
    </tr>
    <tr>
      <th>poolcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>pooltypeid7</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>propertycountylandusecode</th>
      <td>010D</td>
      <td>010D</td>
    </tr>
    <tr>
      <th>propertylandusetypeid</th>
      <td>269.0</td>
      <td>269.0</td>
    </tr>
    <tr>
      <th>propertyzoningdesc</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>rawcensustractandblock</th>
      <td>60378002.041</td>
      <td>60378002.041</td>
    </tr>
    <tr>
      <th>regionidcity</th>
      <td>37688.0</td>
      <td>37688.0</td>
    </tr>
    <tr>
      <th>regionidcounty</th>
      <td>3101.0</td>
      <td>3101.0</td>
    </tr>
    <tr>
      <th>regionidneighborhood</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>regionidzip</th>
      <td>96337.0</td>
      <td>96337.0</td>
    </tr>
    <tr>
      <th>roomcnt</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>threequarterbathnbr</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>unitcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>yearbuilt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>numberofstories</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>structuretaxvaluedollarcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>taxvaluedollarcnt</th>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>assessmentyear</th>
      <td>2015.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>landtaxvaluedollarcnt</th>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>taxamount</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>censustractandblock</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
compare_properties_datasets(properties_2016, properties_2017, 10759547)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>properties_2016</th>
      <th>properties_2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>parcelid</th>
      <td>10754147</td>
      <td>10754147</td>
    </tr>
    <tr>
      <th>airconditioningtypeid</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bathroomcnt</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>bedroomcnt</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>buildingqualitytypeid</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calculatedbathnbr</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedfloor1squarefeet</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>calculatedfinishedsquarefeet</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedsquarefeet12</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedsquarefeet15</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>finishedsquarefeet50</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fips</th>
      <td>6037.0</td>
      <td>6037.0</td>
    </tr>
    <tr>
      <th>fireplacecnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>fullbathcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>garagecarcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>garagetotalsqft</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>heatingorsystemtypeid</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>34144442.0</td>
      <td>34144442.0</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-118654084.0</td>
      <td>-118654084.0</td>
    </tr>
    <tr>
      <th>lotsizesquarefeet</th>
      <td>85768.0</td>
      <td>85768.0</td>
    </tr>
    <tr>
      <th>poolcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>pooltypeid7</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>propertycountylandusecode</th>
      <td>010D</td>
      <td>010D</td>
    </tr>
    <tr>
      <th>propertylandusetypeid</th>
      <td>269.0</td>
      <td>269.0</td>
    </tr>
    <tr>
      <th>propertyzoningdesc</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>rawcensustractandblock</th>
      <td>60378002.041</td>
      <td>60378002.041</td>
    </tr>
    <tr>
      <th>regionidcity</th>
      <td>37688.0</td>
      <td>37688.0</td>
    </tr>
    <tr>
      <th>regionidcounty</th>
      <td>3101.0</td>
      <td>3101.0</td>
    </tr>
    <tr>
      <th>regionidneighborhood</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>regionidzip</th>
      <td>96337.0</td>
      <td>96337.0</td>
    </tr>
    <tr>
      <th>roomcnt</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>threequarterbathnbr</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>unitcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>yearbuilt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>numberofstories</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>structuretaxvaluedollarcnt</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>taxvaluedollarcnt</th>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>assessmentyear</th>
      <td>2015.0</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>landtaxvaluedollarcnt</th>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>taxamount</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>censustractandblock</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Compare completeness of 2017 vs 2016
properties_2017.isnull().mean().sort_values(ascending=False) * 100 - properties_2016.isnull().mean().sort_values(ascending=False) * 100
```




    airconditioningtypeid          -0.128734
    assessmentyear                 -0.284937
    bathroomcnt                    -0.284904
    bedroomcnt                     -0.284904
    buildingqualitytypeid          -0.097380
    calculatedbathnbr              -0.393807
    calculatedfinishedsquarefeet   -0.350661
    censustractandblock            -0.004723
    finishedfloor1squarefeet       -0.034872
    finishedsquarefeet12           -0.388648
    finishedsquarefeet15            0.020535
    finishedsquarefeet50           -0.034872
    fips                           -0.284904
    fireplacecnt                   -0.016314
    fullbathcnt                    -0.393807
    garagecarcnt                   -0.259311
    garagetotalsqft                -0.259311
    heatingorsystemtypeid          -2.102460
    landtaxvaluedollarcnt          -0.261522
    latitude                       -0.284904
    longitude                      -0.284904
    lotsizesquarefeet              -0.113660
    numberofstories                -0.120829
    parcelid                        0.000000
    poolcnt                        -0.740248
    pooltypeid7                    -0.684573
    propertycountylandusecode      -0.310798
    propertylandusetypeid          -0.284904
    propertyzoningdesc             -0.128701
    rawcensustractandblock         -0.284904
    regionidcity                   -0.024018
    regionidcounty                 -0.284904
    regionidneighborhood           -0.011356
    regionidzip                    -0.042409
    roomcnt                        -0.284937
    structuretaxvaluedollarcnt     -0.285339
    taxamount                      -0.284669
    taxvaluedollarcnt              -0.277501
    threequarterbathnbr            -0.158313
    unitcnt                        -0.118986
    yearbuilt                      -0.405163
    dtype: float64




```python
ids_2016 = set(train_2016['parcelid'])
ids_2017 = set(train_2017['parcelid'])

common_ids = ids_2016.intersection(ids_2017)

print(f"Number of parcelids appearing in both datasets: {len(common_ids)}")

counts_2016 = train_2016[train_2016['parcelid'].isin(common_ids)]['parcelid'].value_counts()
counts_2017 = train_2017[train_2017['parcelid'].isin(common_ids)]['parcelid'].value_counts()

parcelid_counts = pd.DataFrame({
    'count_2016': counts_2016,
    'count_2017': counts_2017
}).fillna(0).astype(int)

parcelid_counts.value_counts()
```

    Number of parcelids appearing in both datasets: 2354
    




    count_2016  count_2017
    1           1             2349
    2           1                3
    1           2                2
    Name: count, dtype: int64



**For the purposes of this project, I will try to keep things simple.** 

We will use the 2016 properties data to predict the logerror in 2016 (or **average log error** if a property appears more than once) .

We will discard the 2017 data given that the train data only represents 9/12 months, and the properties data is almost identical. 

We are operating on the basis that log error, and property features are not expected to change meaningfully in our short time period, we will document this as a model assumption. 

Our new objective is therefore to predict the average log error of a Zestimate for a given property which is transacted at least once within 2016, based on the features we engineer. 

 

We will begin to clean our data as follows:

1. We will remove unncessary columns. Comments indicate why this column has been selected for removal with reference to the data dictionary.
2. Fill missing values. Comments indicate why missing values were filled by their respective method. 


```python
properties_2016.propertycountylandusecode.value_counts()
```




    propertycountylandusecode
    0100    1153896
    122      522145
    0101     247494
    010C     225410
    1111     126491
             ...   
    030B          1
    01MC          1
    12T4          1
    5011          1
    3414          1
    Name: count, Length: 240, dtype: int64




```python
properties_2016.regionidneighborhood.value_counts()
```




    regionidneighborhood
    118208.0    32267
    268496.0    23186
    48570.0     21186
    27080.0     18891
    37739.0     18645
                ...  
    764146.0        4
    764092.0        3
    275855.0        3
    275287.0        3
    273552.0        1
    Name: count, Length: 528, dtype: int64




```python
#remove
drop_cols = ['calculatedbathnbr', #identical to bathroomcnt
             'finishedsquarefeet50', #identical to finishedfloor1squarefeet 
             'fullbathcnt', #This counts number of full bathrooms (sink, shower + bathtub, and toilet) present in home but we already have a feature in bathroomcnt to reflect full bathrooms and water closets 
             'threequarterbathnbr', #see above - this counts number of 3/4 bathrooms in house (shower + sink + toilet) wer already have a feature to convey this
             'propertycountylandusecode', #misc alphamnumeric values which we have no mapping for many have leading 0's
             'censustractandblock', #Census tract and block ID combined, once again we have no mapping for this
             'rawcensustractandblock', #identical to censustractandblock
             'propertyzoningdesc', #misc alphamnumeric values which we have no mapping for
             'regionidneighborhood', #Very to regionidzip but less complete
             'latitude', #time-costly to engineer - could use distance based approach (e.g. distance from centre)
             'longitude', #time-costly to engineer - could use distance based approach (e.g. distance from centre)
             'roomcnt', #Values overlap with and sometimes contradict bedroomcnt 
             'unitcnt', #Similar to number of stories, some values contradict
             'taxvaluedollarcnt', #structuretaxvaluedollarcnt + landtaxvaluedollarcnt
             'finishedsquarefeet12', #almost identical to calculatedfinishedsquarefeet 
             'fips' #identical to regionidcounty
            ]

# Drop the columns
properties_2016 = properties_2016.drop(columns=drop_cols, errors="ignore")

print(f"Properties 2016 shape: {properties_2016.shape}")
```

    Properties 2016 shape: (2985217, 25)
    


```python
train_2016.columns
```




    Index(['parcelid', 'logerror', 'transactiondate'], dtype='object')




```python
properties_2017.columns
```




    Index(['parcelid', 'airconditioningtypeid', 'bathroomcnt', 'bedroomcnt',
           'buildingqualitytypeid', 'calculatedbathnbr',
           'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
           'finishedsquarefeet12', 'finishedsquarefeet15', 'finishedsquarefeet50',
           'fips', 'fireplacecnt', 'fullbathcnt', 'garagecarcnt',
           'garagetotalsqft', 'heatingorsystemtypeid', 'latitude', 'longitude',
           'lotsizesquarefeet', 'poolcnt', 'pooltypeid7',
           'propertycountylandusecode', 'propertylandusetypeid',
           'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
           'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
           'threequarterbathnbr', 'unitcnt', 'yearbuilt', 'numberofstories',
           'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',
           'landtaxvaluedollarcnt', 'taxamount', 'censustractandblock'],
          dtype='object')




```python
#Engineer pooltypeid7 to reflect the fact that it represents pools with hot tubs

properties_2016.loc[properties_2016['poolcnt'] == 1, 'pooltypeid7'] = properties_2016.loc[properties_2016['poolcnt'] == 1, 'pooltypeid7'].apply(lambda x: 0 if x == 1 else 1)

properties_2016.rename(columns={'pooltypeid7': 'hottub'}, inplace=True)
```


```python
fill_zero = ['fireplacecnt', 'garagecarcnt', 'garagetotalsqft', 'poolcnt', 'hottub']

for col in fill_zero:
    properties_2016[col] = properties_2016[col].fillna(0)
```


```python
fill_median = ['bathroomcnt', 'bedroomcnt', 'numberofstories', 'buildingqualitytypeid', 'finishedfloor1squarefeet', 
               'calculatedfinishedsquarefeet','finishedsquarefeet15', 'lotsizesquarefeet', 'taxamount', 
               'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'yearbuilt']

for col in fill_median:
    properties_2016[col] = properties_2016[col].fillna(properties_2016[col].median())
```


```python
fill_mode = ['airconditioningtypeid', 'heatingorsystemtypeid', 'propertylandusetypeid', 'regionidcity', 
             'regionidcounty', 'regionidzip', 'assessmentyear']

for col in fill_mode:
    properties_2016[col] = properties_2016[col].fillna(properties_2016[col].mode()[0])
```


```python
#Verify that no more null values remain
print(properties_2016.isnull().sum().sum())
```

    0
    


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Define zoomed-in x-axis limits (adjust as needed)
x_min, x_max = -0.5, 0.5

# Plot distributions zoomed in
plt.figure(figsize=(10,6))
sns.kdeplot(train_2016['logerror'], label='train_2016', fill=True)
plt.title('Distribution of logerror in train_2016 (zoomed in)')
plt.xlabel('logerror')
plt.ylabel('Density')
plt.xlim(x_min, x_max)
plt.legend()
plt.show()
```


    
![png](output_41_0.png)
    


We will retain the outliers above as our objective is to predict the log error of Zestimates, and the features within these are likely to be the most predictive in terms of what is driving log error. 


```python
# Define the threshold range
lower, upper = -0.5, 0.5

outside_2016 = ((train_2016['logerror'] < lower) | (train_2016['logerror'] > upper)).sum()
total_2016 = len(train_2016)
percentage_2016 = (outside_2016 / total_2016) * 100
print(f"Train 2016: {percentage_2016:.2f}% of logerror values are outside [-0.4, 0.4]")
```

    Train 2016: 1.39% of logerror values are outside [-0.4, 0.4]
    


```python
def merge_train_properties(train_df, properties_df):
    """
    :param train_df, properties_df: DataFrames to merge
    :return: merged_df, merged df 
    """
    print(f"Merging train data with properties data")
    merged_df = train_df.merge(properties_df, on='parcelid', how='left')
    print(f"Merged dataset shape: {merged_df.shape}")
    return merged_df

# Merge train datasets with properties datasets
train_properties = merge_train_properties(train_2016, properties_2016)
```

    Merging train data with properties data
    Merged dataset shape: (90275, 27)
    


```python
#train_properties['transactiondate'] = pd.to_datetime(train_properties['transactiondate'])
#train_properties['transactiondate'] = train_properties['transactiondate'].dt.strftime('%Y%m').astype(int)
```


```python
train_properties.shape
```




    (90275, 27)




```python
from sklearn.model_selection import train_test_split

# Step 2: Randomly split 10% as holdout set
data_remainder, holdout_final = train_test_split(train_properties, test_size=0.10, random_state=42)

# Step 3: From the remaining 90%, split 20% as test, and 80% as train
train_final, test_final = train_test_split(data_remainder, test_size=0.20, random_state=42)

# Step 4: Drop 'parcelid' column if needed
train_final = train_final.drop(columns=['parcelid'], errors='ignore')
test_final = test_final.drop(columns=['parcelid'], errors='ignore')
holdout_final = holdout_final.drop(columns=['parcelid'], errors='ignore')
```


```python
train_final.to_excel('train_final_v1.xlsx', index=False)
test_final.to_excel('test_final_v1.xlsx', index=False)
holdout_final.to_excel('holdout_final_v1.xlsx', index=False)
```

### Load saved data


```python
import os
os.chdir('C:\\Users\\bayle\\Directory')
```


```python
import pandas as pd

train_final = pd.read_excel('train_final_v1.xlsx')
test_final = pd.read_excel('test_final_v1.xlsx')
holdout_final = pd.read_excel('holdout_final_v1.xlsx')

print("DataFrames loaded")
```

    DataFrames loaded
    


```python
feature_names = ['parcelid', 'logerror', 'transactiondate', 'airconditioningtypeid', 'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet15', 'fireplacecnt', 'garagecarcnt', 'garagetotalsqft',
 'heatingorsystemtypeid', 'lotsizesquarefeet', 'poolcnt', 'hottub', 'propertylandusetypeid', 'regionidcity', 'regionidcounty', 'regionidzip',
 'yearbuilt', 'numberofstories', 'structuretaxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt', 'taxamount']
```


```python
X_train = train_final.drop(columns=["logerror"])
y_train = train_final["logerror"].values

X_test = test_final.drop(columns=["logerror"])
y_test = test_final["logerror"].values

X_holdout = holdout_final.drop(columns=["logerror"])
y_holdout = holdout_final["logerror"].values
```

# Model Training and Evaluation

We will take a simple approach that compares five candidate model architectures:
1. Linear Regression (Baseline model)
2. Gradient-boosted Trees
3. XGBoost
4. Simple Neural Network
5. Neural network with regularisation techniques

## Evaluation Metrics

To assess model performance, we will use the following regression metrics: **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R-squared ($R^2$)**.

### Mean Squared Error (MSE)

The **Mean Squared Error** measures the average of the squared differences between actual and predicted values:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Interpretation:**  
  Lower MSE indicates better predictive accuracy. It penalizes larger errors more heavily due to the squaring, making it sensitive to outliers.

### Root Mean Squared Error (RMSE)

The **Root Mean Squared Error** is simply the square root of MSE:

$$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$$

**Interpretation:**  
  RMSE represents the **average magnitude of prediction error**, measured in the same units as the target variable. A lower RMSE indicates that the model's predictions are, on average, closer to the true values.

### R-squared ($R^2$)

The **R-squared** metric measures the proportion of variance in the target variable that is explained by the model:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where $\bar{y}$ is the mean of the true values.

**Interpretation:**  
  $R^2$ ranges from 0 to 1 (or can be negative if the model is worse than predicting the mean).  
  - $R^2 = 1$ → perfect prediction  
  - $R^2 = 0$ → model predicts no better than the mean  
  - $R^2 < 0$ → model performs worse than the mean baseline

A higher $R^2$ indicates a better fit, meaning more of the variance in the target is captured by the model.

These metrics provide a comprehensive view of model accuracy and are used throughout to compare models on a consistent basis.


## Linear Regression (Baseline model)

As a starting point, we will use a **Linear Regression** model to establish baseline performance. Linear regression assumes a linear relationship between the input features and the target variable. The model predicts the target $y$ as a weighted sum of the input features $\mathbf{x}$:

$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p = \beta_0 + \sum_{j=1}^p \beta_j x_j$

where:  
$\hat{y}$ is the predicted value,  
$\beta_0$ is the intercept term,  
$\beta_j$ are the coefficients (weights) for each feature $x_j$,  
$p$ is the number of features.

This provides a simple but important benchmark to compare more complex models against.

The model parameters $\boldsymbol{\beta}$ = ($\beta_0$, $\beta_1$, $\ldots$, $\beta_p$) are estimated by minimising the **Mean Squared Error (MSE)** loss:

$\mathcal{L}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2 = \frac{1}{n} \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij} \right)^2$

where $n$ is the number of samples.

Minimizing this loss finds the best linear fit to the data in a least squares sense.


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

linreg_predictions = linreg.predict(X_test_scaled)

mse = mean_squared_error(y_test, linreg_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, linreg_predictions)

print(f"Linear Regression Mean Squared Error (MSE): {mse:.4f}")
print(f"Linear Regression Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Linear Regression R-squared (R2): {r2:.4f}")

import pickle

with open('linreg_model_v1.pkl', 'wb') as f:
    pickle.dump(linreg, f)
```

    Linear Regression Mean Squared Error (MSE): 0.0224
    Linear Regression Root Mean Squared Error (RMSE): 0.1496
    Linear Regression R-squared (R2): 0.0049
    


```python
import pickle

with open('linreg_model_v1.pkl', 'rb') as f:
    linreg = pickle.load(f)

X_holdout_scaled = scaler.transform(X_holdout)

holdout_predictions = linreg.predict(X_holdout_scaled)

mse = mean_squared_error(y_holdout, holdout_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_holdout, holdout_predictions)

# Output results
evaluation_results_linreg = [
    f"mse: {mse:.4f}",
    f"rmse: {rmse:.4f}",
    f"r-squared: {r2:.4f}"
]

print(evaluation_results_linreg)
```

    ['mse: 0.0271', 'rmse: 0.1647', 'r-squared: 0.0024']
    


```python
print(linreg.coef_)
```

    [ 8.10648213e-04  6.14782867e-04 -2.34554001e-03 -3.22802432e-06
      4.09844965e-04  7.45339498e-04  1.00285517e-02 -8.09418316e-04
     -4.08211415e-04  1.69230504e-04 -6.17457905e-04 -2.43621139e-03
      5.08525356e-04 -3.59410779e-03 -2.03670639e-03  3.51754128e-04
     -7.22549624e-04  1.07574235e-03 -1.77174191e-03  5.78115604e-04
      3.68396057e-04  1.11677055e-02  8.67361738e-19  9.51946044e-03
     -2.22817361e-02]
    

## Gradient Boosted Trees

Gradient Boosted Trees (GBTs) are an ensemble learning method that builds a strong predictive model by **sequentially combining many weak learners**—in this case, decision trees. Each tree is trained to correct the errors made by the ensemble of previous trees.

The prediction $\hat{y}_i$ for sample $i$ is the sum of predictions from $M$ individual trees:
$\hat{y}_i = \sum_{m=1}^M f_m(\mathbf{x}_i)$

where each $f_m$ is a decision tree, and $\mathbf{x}_i$ is the feature vector for sample $i$.

Gradient boosting minimises a differentiable loss function $\mathcal{L}$ by adding trees that predict the **negative gradient (residual errors)** of the loss at each iteration:

$\mathcal{L} = \sum_{i=1}^n l(y_i, \hat{y}_i)$

where $l$ is the loss for a single example, such as the squared error:

$l(y_i, \hat{y}_i) = \left(y_i - \hat{y}_i\right)^2$

At each boosting step $m$, the new tree $f_m$ is fitted to the residuals $r_i^{(m)}$ where:

$r_i^{(m)} = - \left[ \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i} \right]_{\hat{y}_i = \hat{y}_i^{(m-1)}}$

are the negative gradients calculated from the previous prediction $\hat{y}_i^{(m-1)}$.

By sequentially adding trees to correct errors, the model improves predictions iteratively.

For more detail see: Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232. Link: https://jerryfriedman.su.domains/ftp/trebst.pdf



```python
from sklearn.ensemble import GradientBoostingRegressor

gbt = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

gbt.fit(X_train, y_train)
predictions = gbt.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

import pickle

with open('gbt_model_v1.pkl', 'wb') as f:
    pickle.dump(gbt, f)
```

    Mean Squared Error (MSE): 0.0223
    Root Mean Squared Error (RMSE): 0.1494
    R-squared (R2): 0.0079
    


```python
with open('gbt_model_v1.pkl', 'rb') as f:
    gbt = pickle.load(f)

holdout_predictions = gbt.predict(X_holdout)

mse = mean_squared_error(y_holdout, holdout_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_holdout, holdout_predictions)

# Output results
evaluation_results_gbt = [
    f"mse: {mse:.4f}",
    f"rmse: {rmse:.4f}",
    f"r-squared: {r2:.4f}"
]

print(evaluation_results_gbt)
```

    ['mse: 0.0275', 'rmse: 0.1658', 'r-squared: -0.0115']
    

## XGBoost: eXtreme Gradient Boosting

**XGBoost** is an optimized implementation of Gradient Boosted Trees (GBTs), designed for speed and performance. It extends the basic gradient boosting framework with several enhancements such as regularization, parallelisation, and improved handling of sparse data.


XGBoost builds trees **sequentially**, just like standard Gradient Boosted Trees (see above). Each tree attempts to correct the residuals (errors) from the sum of the previous trees:

$\hat{y}_i = \sum_{m=1}^M f_m(\mathbf{x}_i), \quad f_m \in \mathcal{F}$

where $\mathcal{F}$ is the space of regression trees.

However, XGBoost improves on GBTs with:

- **Second-order optimization** using both gradients (first order derivatives) and Hessians (second order derivatives)
- **L1 and L2 regularization** on leaf weights
- **Pruning** only removes splits if the loss reduction (gain) is less than a threshold
- **Feature subsampling** for speed and generalisation
- **Built-in handling of missing values**
- **System and Engineering optimisations** (e.g. parallel computing support)

At iteration $t$, XGBoost minimizes a regularized objective:

$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$

where $l$ is the loss (e.g., squared error), and the regularization term $\Omega$ penalizes model complexity:

$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$

- $T$ = number of leaves in the tree  
- $w_j$ = score (weight) of leaf $j$  
- $\gamma$, $\lambda$ = regularization parameters  

This helps control overfitting and encourages simpler models.

For more detail see: Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM. Link: https://arxiv.org/pdf/1603.02754




```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbosity=1, 
    objective='reg:squarederror'
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, xgb_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, xgb_predictions)

print(f"XGBoost Mean Squared Error (MSE): {mse:.4f}")
print(f"XGBoost Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"XGBoost R-squared (R2): {r2:.4f}")

xgb_model.save_model('xgboost_model.json')
```

    XGBoost Mean Squared Error (MSE): 0.0224
    XGBoost Root Mean Squared Error (RMSE): 0.1496
    XGBoost R-squared (R2): 0.0059
    


```python
xgb_model.load_model('xgboost_model.json')

holdout_predictions = xgb_model.predict(X_holdout)

mse = mean_squared_error(y_holdout, holdout_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_holdout, holdout_predictions)

# Output results
evaluation_results_xgb = [
    f"mse: {mse:.4f}",
    f"rmse: {rmse:.4f}",
    f"r-squared: {r2:.4f}"
]

print(evaluation_results_xgb)
```

    ['mse: 0.0269', 'rmse: 0.1641', 'r-squared: 0.0100']
    

## Simple Neural Network 

Feedforward Neural Networks are comprised of multiple layers. 

Each layer in a neural network is made up of multiple neurons. A neuron performs a simple operation:

$z = w^T x + b$

Where:

$x$ = input vector

$w$ = weights

$b$ = bias

The result $z$ is passed through an **activation function** to introduce nonlinearity.


Data flows forward through the network:

Input layer: Receives the raw input

Hidden layers: Transform the input through weights and activation functions (e.g. ReLU)

Output layer: Produces a prediction (e.g., a single continuous value for regression)

Each layer learns to extract increasingly abstract representations of the input.

Neural networks learn by minimizing a loss function, such as Mean Squared Error (MSE) for regression. The process:

Forward pass: Make predictions $\hat{y}$

Loss computation: Measure how far predictions are from the true labels

Backward pass (backpropagation): Compute gradients of the loss with respect to each weight

Weight update: Use an optimizer (like Adam) to adjust weights and reduce the loss

This is repeated over many epochs (one complete iteration through the training set) until the model converges.

Layer stacking allows deep feature transformations.

Training with large data and efficient optimization leads to models that generalize well to unseen examples.

For detail see: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533–536. Link: https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

FNN = SimpleRegressor()

criterion = nn.MSELoss()
optimizer = optim.Adam(FNN.parameters(), lr=0.001)

epochs = 50
batch_size = 1024
patience = 10  # Early stopping patience
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
best_model_state = None

for epoch in range(epochs):
    FNN.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = FNN(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= X_train_tensor.size(0)
    train_losses.append(epoch_loss)

    FNN.eval()
    with torch.no_grad():
        val_outputs = FNN(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = FNN.state_dict()  # Save the best model state
        torch.save(best_model_state, 'best_model_FNN.pth')  # Save model to disk
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

FNN.load_state_dict(torch.load('best_model_FNN.pth'))

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

    Epoch 1/50 | Train Loss: 0.0283 | Val Loss: 0.0226
    Epoch 2/50 | Train Loss: 0.0266 | Val Loss: 0.0226
    Epoch 3/50 | Train Loss: 0.0265 | Val Loss: 0.0225
    Epoch 4/50 | Train Loss: 0.0265 | Val Loss: 0.0226
    Epoch 5/50 | Train Loss: 0.0265 | Val Loss: 0.0228
    Epoch 6/50 | Train Loss: 0.0263 | Val Loss: 0.0227
    Epoch 7/50 | Train Loss: 0.0263 | Val Loss: 0.0224
    Epoch 8/50 | Train Loss: 0.0262 | Val Loss: 0.0225
    Epoch 9/50 | Train Loss: 0.0262 | Val Loss: 0.0225
    Epoch 10/50 | Train Loss: 0.0261 | Val Loss: 0.0225
    Epoch 11/50 | Train Loss: 0.0260 | Val Loss: 0.0227
    Epoch 12/50 | Train Loss: 0.0261 | Val Loss: 0.0223
    Epoch 13/50 | Train Loss: 0.0260 | Val Loss: 0.0225
    Epoch 14/50 | Train Loss: 0.0259 | Val Loss: 0.0227
    Epoch 15/50 | Train Loss: 0.0259 | Val Loss: 0.0224
    Epoch 16/50 | Train Loss: 0.0259 | Val Loss: 0.0226
    Epoch 17/50 | Train Loss: 0.0258 | Val Loss: 0.0225
    Epoch 18/50 | Train Loss: 0.0258 | Val Loss: 0.0226
    Epoch 19/50 | Train Loss: 0.0258 | Val Loss: 0.0225
    Epoch 20/50 | Train Loss: 0.0258 | Val Loss: 0.0226
    Epoch 21/50 | Train Loss: 0.0258 | Val Loss: 0.0227
    Epoch 22/50 | Train Loss: 0.0257 | Val Loss: 0.0225
    Early stopping triggered at epoch 22
    


    
![png](output_67_1.png)
    



```python
import torch
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X_holdout_scaled_tensor = torch.tensor(scaler.transform(X_holdout), dtype=torch.float32)
y_holdout_tensor = torch.tensor(y_holdout.reshape(-1, 1), dtype=torch.float32)

class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

FNN = SimpleRegressor()
FNN.load_state_dict(torch.load('best_model_FNN.pth'))
FNN.eval()

with torch.no_grad():
    holdout_pred_tensor = FNN(X_holdout_scaled_tensor)
    holdout_pred = holdout_pred_tensor.numpy().flatten()

mse = mean_squared_error(y_holdout, holdout_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_holdout, holdout_pred)

evaluation_results_fnn = [
    f"mse: {mse:.4f}",
    f"rmse: {rmse:.4f}",
    f"r-squared: {r2:.4f}"
]

print(evaluation_results_fnn)
```

    ['mse: 0.0270', 'rmse: 0.1644', 'r-squared: 0.0053']
    

## Regularization Techniques Used in `Deep_FNN`

To improve generalization and stability of training, several regularization strategies were incorporated into the `Deep_FNN` model.

---

### 1. **Batch Normalization**

Batch Normalization normalizes each feature in the mini-batch such that it has a $\mu$ = 0, $\sigma$ = 1 to reduce internal covariate shift and accelerate training:

$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i,$

$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2,$

$\hat{x}_i$ = $\frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

$y_i = \gamma \hat{x}_i + \beta$

- $x_i$: input activation  
- $\mu_B, \sigma_B^2$: batch mean and variance  
- $\gamma, \beta$: learnable scale and shift parameters (initially set to $\gamma$ = 1, $\beta$ = 0 and updated via backpropogation) 
- $\epsilon$: small constant for numerical stability

For detail see: Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. Proceedings of the 32nd International Conference on Machine Learning (ICML), 448–456. Link: https://arxiv.org/pdf/1502.03167

---

### 2. **Dropout**

Dropout helps prevent overfitting by randomly deactivating neurons during training:

$$
\tilde{h}_i = h_i \cdot z_i, \quad z_i \sim \text{Bernoulli}(p)
$$

- $h_i$: neuron activation  
- $z_i$: dropout mask (1 with probability \( p \), otherwise 0)  
- $p$: keep probability (e.g., 0.7 when dropout rate = 0.3)

For detail see: Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56), 1929–1958. Link: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

---

### 3. **Gradient Clipping**

Gradient Clipping ensures the norm of the gradient vector does not explode by enforcing a threshold:

$$
\text{If } \|g\| > \theta, \quad g \leftarrow \frac{\theta}{\|g\|} \cdot g
$$

- $g$: gradient vector  
- $\theta$: maximum norm threshold (e.g., 1.0)

For detail see Section 3.2/3.3: Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. Proceedings of the 30th International Conference on Machine Learning (ICML), 1310–1318. Link: https://arxiv.org/pdf/1211.5063

---

### 4. **Learning Rate Scheduling (ReduceLROnPlateau)**

The learning rate is reduced when validation loss plateaus:

$$
\text{If no improvement for } p \text{ epochs:} \quad \text{lr}_{\text{new}} = \text{lr}_{\text{old}} \times \text{factor}
$$

- $p$: patience (number of stagnant epochs, e.g., 3)  
- $\text{factor}$: decay factor (e.g., 0.5)  



```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

class DeepRegressor(nn.Module):
    def __init__(self):
        super(DeepRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

Deep_FNN = DeepRegressor()

criterion = nn.MSELoss()
optimizer = optim.Adam(Deep_FNN.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

epochs = 100
batch_size = 1024
patience = 10

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

train_losses = []
val_losses = []

for epoch in range(epochs):
    Deep_FNN.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = Deep_FNN(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Deep_FNN.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= X_train_tensor.size(0)
    train_losses.append(epoch_loss)

    Deep_FNN.eval()
    with torch.no_grad():
        val_outputs = Deep_FNN(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)

        # For real-world scale metrics
        val_preds = scaler_y.inverse_transform(val_outputs.cpu().numpy())
        val_true = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())
        val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
        val_r2 = r2_score(val_true, val_preds)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = Deep_FNN.state_dict()
        torch.save(best_model_state, 'best_deep_fnn.pth') 
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

Deep_FNN.load_state_dict(torch.load('best_deep_fnn.pth'))

Deep_FNN.eval()
with torch.no_grad():
    final_preds = Deep_FNN(X_test_tensor)
    final_preds_rescaled = scaler_y.inverse_transform(final_preds.numpy())
    y_test_rescaled = scaler_y.inverse_transform(y_test_tensor.numpy())

    final_mse = mean_squared_error(y_test_rescaled, final_preds_rescaled)
    final_rmse = np.sqrt(final_mse)
    final_r2 = r2_score(y_test_rescaled, final_preds_rescaled)

print(f"\nFinal Test Metrics:\nMSE: {final_mse:.4f} | RMSE: {final_rmse:.4f} | R²: {final_r2:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Learning Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

    Epoch 1/100 | Train Loss: 1.067818 | Val Loss: 0.844046 | RMSE: 0.1499 | R²: 0.0007
    Epoch 2/100 | Train Loss: 1.015030 | Val Loss: 0.842452 | RMSE: 0.1498 | R²: 0.0026
    Epoch 3/100 | Train Loss: 1.002122 | Val Loss: 0.841105 | RMSE: 0.1497 | R²: 0.0042
    Epoch 4/100 | Train Loss: 0.998155 | Val Loss: 0.841611 | RMSE: 0.1497 | R²: 0.0036
    Epoch 5/100 | Train Loss: 0.996266 | Val Loss: 0.841603 | RMSE: 0.1497 | R²: 0.0036
    Epoch 6/100 | Train Loss: 0.995794 | Val Loss: 0.840957 | RMSE: 0.1497 | R²: 0.0044
    Epoch 7/100 | Train Loss: 0.995263 | Val Loss: 0.840264 | RMSE: 0.1496 | R²: 0.0052
    Epoch 8/100 | Train Loss: 0.994185 | Val Loss: 0.840710 | RMSE: 0.1496 | R²: 0.0047
    Epoch 9/100 | Train Loss: 0.992865 | Val Loss: 0.841048 | RMSE: 0.1497 | R²: 0.0043
    Epoch 10/100 | Train Loss: 0.993420 | Val Loss: 0.840102 | RMSE: 0.1496 | R²: 0.0054
    Epoch 11/100 | Train Loss: 0.992658 | Val Loss: 0.840214 | RMSE: 0.1496 | R²: 0.0053
    Epoch 12/100 | Train Loss: 0.991223 | Val Loss: 0.840879 | RMSE: 0.1497 | R²: 0.0045
    Epoch 13/100 | Train Loss: 0.991940 | Val Loss: 0.841400 | RMSE: 0.1497 | R²: 0.0039
    Epoch 14/100 | Train Loss: 0.991594 | Val Loss: 0.840311 | RMSE: 0.1496 | R²: 0.0052
    Epoch 15/100 | Train Loss: 0.990148 | Val Loss: 0.840731 | RMSE: 0.1496 | R²: 0.0047
    Epoch 16/100 | Train Loss: 0.989853 | Val Loss: 0.840258 | RMSE: 0.1496 | R²: 0.0052
    Epoch 17/100 | Train Loss: 0.989754 | Val Loss: 0.841675 | RMSE: 0.1497 | R²: 0.0036
    Epoch 18/100 | Train Loss: 0.988953 | Val Loss: 0.841830 | RMSE: 0.1497 | R²: 0.0034
    Epoch 19/100 | Train Loss: 0.988113 | Val Loss: 0.840872 | RMSE: 0.1497 | R²: 0.0045
    Epoch 20/100 | Train Loss: 0.988238 | Val Loss: 0.841186 | RMSE: 0.1497 | R²: 0.0041
    Early stopping triggered at epoch 20
    
    Final Test Metrics:
    MSE: 0.0224 | RMSE: 0.1496 | R²: 0.0054
    


    
![png](output_70_1.png)
    



```python
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_holdout_scaled = scaler_y.transform(y_holdout.reshape(-1, 1))

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_holdout_scaled_tensor = torch.tensor(scaler.transform(X_holdout), dtype=torch.float32)
y_holdout_tensor = torch.tensor(y_holdout_scaled, dtype=torch.float32)

class DeepRegressor(nn.Module):
    def __init__(self):
        super(DeepRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

Deep_FNN = DeepRegressor()
Deep_FNN.load_state_dict(torch.load('best_deep_fnn.pth'))
Deep_FNN.eval()

with torch.no_grad():
    y_pred_tensor = Deep_FNN(X_holdout_scaled_tensor)
    y_pred_scaled = y_pred_tensor.numpy().flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_holdout, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_holdout, y_pred)

evaluation_results_deepfnn = [
    f"mse: {mse:.4f}",
    f"rmse: {rmse:.4f}",
    f"r-squared: {r2:.4f}"
]

print(evaluation_results_deepfnn)
```

    ['mse: 0.0270', 'rmse: 0.1643', 'r-squared: 0.0074']
    

## Hyperparameter tuning

Typically, we would tune the hyperparameters of all candidate models to perform the fairest evaluation of each. However due to compute restraints here, I will tune the hyperparameters for our best performing model: XGBoost.


```python
from sklearn.model_selection import RandomizedSearchCV

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1],
    'colsample_bytree': [0.7, 0.8, 0.9, 1],
    'reg_alpha': [0, 0.05, 0.1],
    'reg_lambda': [1, 1.5, 2],
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=100,
    
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best params:", random_search.best_params_)
print("Best CV R2:", random_search.best_score_)
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    Best params: {'subsample': 0.7, 'reg_lambda': 2, 'reg_alpha': 0.1, 'n_estimators': 300, 'min_child_weight': 2, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.7}
    Best CV R2: 0.011324387369790846
    

# Therefore, our best model is as follows:


```python
from xgboost import XGBRegressor

best_xgb_model = XGBRegressor(
    subsample=0.7,
    reg_lambda=2,
    reg_alpha=0.1,
    n_estimators=300,
    min_child_weight=2,
    learning_rate=0.01,
    max_depth=5,
    gamma=0,
    colsample_bytree=0.7,
    random_state=42,
    verbosity=1, 
)

best_xgb_model.fit(X_train, y_train)

best_xgb_predictions = best_xgb_model.predict(X_test)

mse = mean_squared_error(y_test, best_xgb_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, best_xgb_predictions)

print(f"Best XGBoost Mean Squared Error (MSE): {mse:.4f}")
print(f"Best XGBoost Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Best XGBoost R-squared (R2): {r2:.4f}")

best_xgb_model.save_model('best_xgboost_model.json')
```

    Best XGBoost Mean Squared Error (MSE): 0.0223
    Best XGBoost Root Mean Squared Error (RMSE): 0.1493
    Best XGBoost R-squared (R2): 0.0088
    


```python
best_xgb_model.load_model('best_xgboost_model.json')

holdout_predictions = best_xgb_model.predict(X_holdout)

mse = mean_squared_error(y_holdout, holdout_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_holdout, holdout_predictions)

# Output results
evaluation_results_best_xgb = [
    f"mse: {mse:.4f}",
    f"rmse: {rmse:.4f}",
    f"r-squared: {r2:.4f}"
]

print(evaluation_results_best_xgb)
```

    ['mse: 0.0269', 'rmse: 0.1640', 'r-squared: 0.0112']
    

# Conclusion:


```python
print(evaluation_results_linreg)
```

    ['mse: 0.0271', 'rmse: 0.1647', 'r-squared: 0.0024']
    


```python
print(evaluation_results_gbt)
```

    ['mse: 0.0275', 'rmse: 0.1658', 'r-squared: -0.0115']
    


```python
print(evaluation_results_xgb)
```

    ['mse: 0.0269', 'rmse: 0.1641', 'r-squared: 0.0100']
    


```python
print(evaluation_results_fnn)
```

    ['mse: 0.0270', 'rmse: 0.1644', 'r-squared: 0.0053']
    


```python
print(evaluation_results_deepfnn)
```

    ['mse: 0.0270', 'rmse: 0.1643', 'r-squared: 0.0074']
    


```python
print(evaluation_results_best_xgb)
```

    ['mse: 0.0269', 'rmse: 0.1640', 'r-squared: 0.0112']
    

Overall, our best xgb model has the best metrics, as mse and rmse are low, and r-squared is high. 

# Model Explainability

To make our model more explainable we will review: 
1. Feature importance
2. SHAP values 

## Feature Importance
XGBoost measures feature importance by quantifying how much each feature contributes to improving the model during training. It does this based on how often and how effectively a feature is used to split nodes in the boosted trees.

**Gain (Average Gain)**  
   Measures the average improvement in the loss function (reduction in error) brought by splits using the feature.  
   For a split $s$ on feature $f$, the gain is calculated as:  
    $\text{Gain}(s) = \frac{1}{2} \left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right) - \gamma$
    
   where:  
   - $G_L$, $H_L$ = sum of gradients and Hessians in the left child  
   - $G_R$, $H_R$ = sum of gradients and Hessians in the right child  
   - $\lambda$= L2 regularization term on leaf weights  
   - $\gamma$ = complexity penalty for making a split

   The **Gain** for a feature $f$ is the sum or average of gains over all splits using $f$:

   $\text{Gain}(f) = \frac{1}{N_f} \sum_{s \in S_f} \text{Gain}(s)$
   
   where $S_f$ is the set of splits on feature $f$, and $N_f = |S_f|$.

quantifies how much a feature improves model accuracy per split (based on gradients and Hessians), it directly relates to the objective function improvement during training.



```python
importance = best_xgb_model.get_booster().get_score(importance_type='gain') 

total = sum(importance.values())
importance_normalized = {k: v / total for k, v in importance.items()}

sorted_importance = sorted(importance_normalized.items(), key=lambda x: x[1], reverse=True)

for feature, score in sorted_importance:
    print(f"{feature}: {score:.4f}")
```

    lotsizesquarefeet: 0.0588
    regionidcity: 0.0570
    structuretaxvaluedollarcnt: 0.0503
    poolcnt: 0.0492
    landtaxvaluedollarcnt: 0.0492
    regionidcounty: 0.0489
    airconditioningtypeid: 0.0488
    bathroomcnt: 0.0488
    regionidzip: 0.0487
    fireplacecnt: 0.0486
    taxamount: 0.0453
    garagecarcnt: 0.0452
    buildingqualitytypeid: 0.0441
    yearbuilt: 0.0432
    calculatedfinishedsquarefeet: 0.0429
    transactiondate: 0.0366
    garagetotalsqft: 0.0334
    finishedfloor1squarefeet: 0.0331
    bedroomcnt: 0.0323
    heatingorsystemtypeid: 0.0318
    propertylandusetypeid: 0.0282
    finishedsquarefeet15: 0.0279
    hottub: 0.0270
    numberofstories: 0.0207
    

## SHAP Values

SHAP values come from game theory and provide a way to fairly measure how much each feature contributes to a model’s prediction.

- Imagine the prediction as a "payout" to be fairly split among all features.  
- The SHAP value for a feature is its **average contribution** to the prediction across all possible combinations of features.  
- Formally, for feature $i$, the SHAP value $\phi_i$ is calculated by averaging the change in prediction when adding feature $i$ to every subset $S$ of features not containing $i$:

$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S) \right]$

where:  
- $N$ = all features  
- $f_S(x_S)$ = model prediction using features in subset $S$

For detail see: Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30. Link: https://arxiv.org/pdf/1705.07874

### Intepretation

SHAP values **explain individual predictions** by showing how each feature pushes the prediction higher or lower compared to the average prediction.  

The sum of all SHAP values for a sample equals the difference between that sample’s prediction and the average prediction over all samples.  

Positive SHAP values mean the feature increases the prediction, negative means it decreases the prediction.  

By averaging the absolute SHAP values over many samples, you can find which features are generally the most important to the model.

SHAP values provide a fair and consistent way to assign credit to features for each prediction.  

They show both the **direction** and **magnitude** of feature influence.  

This helps understand how the model makes decisions on both a global level (overall feature importance) and a local level (individual prediction explanations).



```python
import shap

explainer = shap.Explainer(best_xgb_model)

shap_values = explainer(X_train)

#shap.summary_plot(shap_values, X_train)

#shap.dependence_plot('airconditioningtypeid', shap_values.values, X_train)

mean_shap = np.mean(shap_values.values, axis=0)

# Create a DataFrame with feature names and mean absolute SHAP values
shap_df = pd.DataFrame({
    'feature': X_train.columns,
    'mean_shap': mean_shap
})

# Sort by descending importance
shap_df_sorted = shap_df.sort_values(by='mean_shap', ascending=False)

# Print the sorted SHAP values
print(shap_df_sorted.reset_index(drop=True).round(5))
```

                             feature  mean_shap
    0                      taxamount    0.00073
    1                      yearbuilt    0.00020
    2                    bathroomcnt    0.00015
    3   calculatedfinishedsquarefeet    0.00007
    4          heatingorsystemtypeid    0.00004
    5                   regionidcity    0.00004
    6          propertylandusetypeid    0.00002
    7                        poolcnt    0.00002
    8                    regionidzip    0.00002
    9                         hottub    0.00000
    10         airconditioningtypeid    0.00000
    11               numberofstories    0.00000
    12                assessmentyear    0.00000
    13                regionidcounty   -0.00000
    14      finishedfloor1squarefeet   -0.00000
    15         buildingqualitytypeid   -0.00000
    16               transactiondate   -0.00001
    17                  garagecarcnt   -0.00001
    18                  fireplacecnt   -0.00001
    19                    bedroomcnt   -0.00002
    20          finishedsquarefeet15   -0.00003
    21               garagetotalsqft   -0.00007
    22             lotsizesquarefeet   -0.00009
    23         landtaxvaluedollarcnt   -0.00049
    24    structuretaxvaluedollarcnt   -0.00053
    


```python
# Choose one instance to explain (e.g., row 0)
instance = X_train.iloc[[100]]  # keep as DataFrame

# Compute SHAP values for the instance
explainer = shap.Explainer(best_xgb_model)
shap_values_single = explainer(instance)

# Get the SHAP values and base value (expected value)
shap_values_array = shap_values_single.values[0]  # shape: (n_features,)
base_value = shap_values_single.base_values[0]    # model's expected value
predicted_value = shap_values_single.data[0].dot(best_xgb_model.feature_importances_)  # or use model.predict()

# Create a DataFrame for all SHAP values of this instance
shap_df_single = pd.DataFrame({
    'feature': X_train.columns,
    'shap_value': shap_values_array
})

# Sort by absolute SHAP importance
shap_df_single['abs_shap'] = np.abs(shap_df_single['shap_value'])
shap_df_single_sorted = shap_df_single.sort_values(by='abs_shap', ascending=False)

# Drop the helper column for clarity
shap_df_single_sorted = shap_df_single_sorted.drop(columns='abs_shap')

# Print the sorted SHAP values
print(shap_df_single_sorted.reset_index(drop=True).round(5))

# Optionally show predicted value breakdown
print(f"\nBase value (expected model output): {base_value:.5f}")
print(f"Predicted value for this instance: {(base_value + shap_values_array.sum()):.5f}")
```

                             feature  shap_value
    0                      taxamount     0.00590
    1          landtaxvaluedollarcnt     0.00202
    2                     bedroomcnt    -0.00136
    3     structuretaxvaluedollarcnt    -0.00130
    4   calculatedfinishedsquarefeet    -0.00129
    5                        poolcnt     0.00077
    6                    bathroomcnt    -0.00073
    7              lotsizesquarefeet    -0.00060
    8                    regionidzip     0.00056
    9          propertylandusetypeid     0.00046
    10                     yearbuilt    -0.00024
    11         heatingorsystemtypeid     0.00021
    12          finishedsquarefeet15    -0.00006
    13               transactiondate    -0.00005
    14                        hottub     0.00005
    15                regionidcounty    -0.00005
    16                  regionidcity    -0.00003
    17                  garagecarcnt    -0.00003
    18               garagetotalsqft    -0.00002
    19         buildingqualitytypeid    -0.00002
    20                  fireplacecnt    -0.00001
    21      finishedfloor1squarefeet    -0.00001
    22               numberofstories     0.00000
    23         airconditioningtypeid    -0.00000
    24                assessmentyear     0.00000
    
    Base value (expected model output): 0.01149
    Predicted value for this instance: 0.01566
    

### Note: This notebook provides a demonstration of how to train and evaluate regression models using a subset of the data. However, deploying a model to production involves many additional considerations beyond model fitting—such as data pipeline design, monitoring for data drift, model versioning, and ensuring reliability, scalability, and maintainability in a real-world environment.
