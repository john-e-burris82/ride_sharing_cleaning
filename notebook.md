# Cleaning Data in Python

👋 Welcome to your workspace! Here, you can write and run Python code and add text in [Markdown](https://www.markdownguide.org/basic-syntax/). Below, we've imported the datasets from the course _Cleaning Data in Python_ as DataFrames as well as the packages used in the course. This is your sandbox environment: analyze the course datasets further, take notes, or experiment with code!


```python
%%capture
# Install fuzzywuzzy
!pip install fuzzywuzzy
```


```python
# Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import missingno as msno
import fuzzywuzzy
import recordlinkage 

# Importing course datasets as DataFrames
ride_sharing = pd.read_csv('datasets/ride_sharing_new.csv', index_col = 'Unnamed: 0')
airlines = pd.read_csv('datasets/airlines_final.csv',  index_col = 'Unnamed: 0')
banking = pd.read_csv('datasets/banking_dirty.csv', index_col = 'Unnamed: 0')
restaurants = pd.read_csv('datasets/restaurants_L2.csv', index_col = 'Unnamed: 0')
restaurants_new = pd.read_csv('datasets/restaurants_L2_dirty.csv', index_col = 'Unnamed: 0')

```


```python
# Ride Sharing Data Types and Count of Non-Null Values

ride_sharing.info()

# Another option to begin exploring the data, uncomment it to check it out

# ride_sharing.dtypes
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25760 entries, 0 to 25759
    Data columns (total 9 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   duration         25760 non-null  object
     1   station_A_id     25760 non-null  int64 
     2   station_A_name   25760 non-null  object
     3   station_B_id     25760 non-null  int64 
     4   station_B_name   25760 non-null  object
     5   bike_id          25760 non-null  int64 
     6   user_type        25760 non-null  int64 
     7   user_birth_year  25760 non-null  int64 
     8   user_gender      25760 non-null  object
    dtypes: int64(5), object(4)
    memory usage: 2.0+ MB



```python
# Get Head of Ride Sharing Data to Ensure it Makes Sense

ride_sharing.head(5)
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
      <th>duration</th>
      <th>station_A_id</th>
      <th>station_A_name</th>
      <th>station_B_id</th>
      <th>station_B_name</th>
      <th>bike_id</th>
      <th>user_type</th>
      <th>user_birth_year</th>
      <th>user_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12 minutes</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>323</td>
      <td>Broadway at Kearny</td>
      <td>5480</td>
      <td>2</td>
      <td>1959</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24 minutes</td>
      <td>3</td>
      <td>Powell St BART Station (Market St at 4th St)</td>
      <td>118</td>
      <td>Eureka Valley Recreation Center</td>
      <td>5193</td>
      <td>2</td>
      <td>1965</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8 minutes</td>
      <td>67</td>
      <td>San Francisco Caltrain Station 2  (Townsend St...</td>
      <td>23</td>
      <td>The Embarcadero at Steuart St</td>
      <td>3652</td>
      <td>3</td>
      <td>1993</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4 minutes</td>
      <td>16</td>
      <td>Steuart St at Market St</td>
      <td>28</td>
      <td>The Embarcadero at Bryant St</td>
      <td>1883</td>
      <td>1</td>
      <td>1979</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11 minutes</td>
      <td>22</td>
      <td>Howard St at Beale St</td>
      <td>350</td>
      <td>8th St at Brannan St</td>
      <td>4626</td>
      <td>2</td>
      <td>1994</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>



### Ride Sharing Review

After exploring the variable data types with .dtypes and printing out the head of the data, there is a few items that need to be addressed.

1. Duration obviously would work better as an integer. Therefore, removing the word minutes and changing it to an int data type is required. We can also rename the column to include (min) in the header, however a well crafted data dictionary should also explain this.
2. Station names could be changed to strings but as I researched more, Pandas will generally always store strings as objects. So I will leave this one alone.
3. User type needs to be changed to categorical.
4. user_birth_year needs to be changed to a date represented by year.
5. Gender could be left alone, or it could be changed to categorical with numerical values 0 and 1 representing the values.


```python
#Lets start with duration, stripping minutes and recategorizing as an integer.

ride_sharing['duration'] = ride_sharing['duration'].str.strip(' minutes')
ride_sharing['duration'] =ride_sharing['duration'].astype('int')

# You can also utilize assert to verify that duration is now an integar. It will return false if it is not and return nothing if true. Uncomment to check it out.

# assert ride_sharing['duration'].dtype == 'int'

ride_sharing.rename(columns = {'duration':'duration_mins'}, inplace = True)

ride_sharing.head(1)
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
      <th>duration_mins</th>
      <th>station_A_id</th>
      <th>station_A_name</th>
      <th>station_B_id</th>
      <th>station_B_name</th>
      <th>bike_id</th>
      <th>user_type</th>
      <th>user_birth_year</th>
      <th>user_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>323</td>
      <td>Broadway at Kearny</td>
      <td>5480</td>
      <td>2</td>
      <td>1959</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>



### Changing user_type to categorical

With the code below, I have changed the variable 'user_type' to a category. I utilized assert to ensure the change occured. I followed this with .describe(). With the last line of code, I am able to see there are 25,760 data points, 3 unique values, value 2 is the most common with a frequency of 12,972. It tells me on the last line of the output that I am looking at the variable 'user_type' and that infact it is an int64. I know I said that I changed it to a categorical variable; however these are represented by int64 unless you utilize the text categories. Either way, a well drafted data dictionary should explain what each category means such as 1 = registered user, 2 = dependent of registered user, and 3 = unregistered user.


```python
# Lets now change 'user_type' to a category.

ride_sharing['user_type'] = ride_sharing['user_type'].astype('category')

# Lets use assert to verify
assert ride_sharing['user_type'].dtype == 'category'

ride_sharing['user_type'].value_counts()

```




    2    12972
    3     6502
    1     6286
    Name: user_type, dtype: int64



### Changing user_birth_year to a Date

We will change the varialbe type to a date with a year format. The quickest way to do this is with pd.to_datetime(df['variable']). You can also specify the format. I utilized format='$Y' which unfortunately will add xxxx-01-01 January 1st to every year. To remove this, we need to add .dt.year to strip January 1st off and leave just the year. However, you will find that when you do this, it turns back to an integer. If you want a date, delete .dt.year off the code below, if not just skip this code as it is already an integer. If anyone has any suggestions, let me know in the comment section.


```python
# We now can change the variable type.
ride_sharing['user_birth_year'] = pd.to_datetime(ride_sharing['user_birth_year'], format='%Y').dt.year

# Check it out
ride_sharing.head(2)
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
      <th>duration_mins</th>
      <th>station_A_id</th>
      <th>station_A_name</th>
      <th>station_B_id</th>
      <th>station_B_name</th>
      <th>bike_id</th>
      <th>user_type</th>
      <th>user_birth_year</th>
      <th>user_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>323</td>
      <td>Broadway at Kearny</td>
      <td>5480</td>
      <td>2</td>
      <td>1959</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>3</td>
      <td>Powell St BART Station (Market St at 4th St)</td>
      <td>118</td>
      <td>Eureka Valley Recreation Center</td>
      <td>5193</td>
      <td>2</td>
      <td>1965</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>



### Factorizing user_gender

The first thing I like to do prior to factorizing is to get a good count of each text category. The .value_counts() will do this nicely. We will actually use it twice just for a sanity check. What we find is there are 19,382 male users in the data set, 6,027 females, and 351 that classified as other or unlisted. For basic exploratory analysis, one really doesn't need to factorize these; however, most prediction models work better with numerical data.


```python
#Convert 'user_gender' to a factor 
ride_sharing['user_gender'].value_counts()
ride_sharing['user_gender'] = pd.factorize(ride_sharing['user_gender'])[0]
ride_sharing['user_gender'] = ride_sharing['user_gender'].astype('category')
ride_sharing['user_gender'].value_counts()
```




    0    19382
    1     6027
    2      351
    Name: user_gender, dtype: int64



### Removing white space
I always like to run the following code to remove any leading and trailing whitespace. This is similar to the trim() function in Excel. We all run into this issue from time to time. So this is actually a good starting point. It is a little advanced and utilizes a lambda function. Not a topic for discussion here.


```python
# Create whitespace trimming function
cols = ride_sharing.select_dtypes('object').columns
ride_sharing[cols] = ride_sharing[cols].apply(lambda x: x.str.strip())
```

### Checking for Duplicates

The last aspect I like to check prior to beginning deeper exploratory analysis is for duplicates. You could check for out of range dates, random categories that do not belong (in reality we already did this), value consistencies, and other constraints. I will provide two methods. The first will check for exact matches across all columns. The second way will require a little more visual comparison; however, the second method will help to catch things if there are spelling errors or other issues. As this is a rather simple data set, I will stick with the first example, trusting it caught all the duplicates and remove those values. After removing the duplicate rows, which should only remove 4 rows, we will check the data to see if it has 25,756 rows. Success, it does.


```python
# Let's check for duplicates that have the same values across all variables

duplicates = ride_sharing.duplicated()
ride_sharing[duplicates]
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
      <th>duration_mins</th>
      <th>station_A_id</th>
      <th>station_A_name</th>
      <th>station_B_id</th>
      <th>station_B_name</th>
      <th>bike_id</th>
      <th>user_type</th>
      <th>user_birth_year</th>
      <th>user_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>604</th>
      <td>9</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>1225</td>
      <td>2</td>
      <td>1993</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15217</th>
      <td>17</td>
      <td>22</td>
      <td>Howard St at Beale St</td>
      <td>102</td>
      <td>Irwin St at 8th St</td>
      <td>492</td>
      <td>3</td>
      <td>1961</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18303</th>
      <td>10</td>
      <td>30</td>
      <td>San Francisco Caltrain (Townsend St at 4th St)</td>
      <td>6</td>
      <td>The Embarcadero at Sansome St</td>
      <td>4442</td>
      <td>1</td>
      <td>1967</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20170</th>
      <td>4</td>
      <td>21</td>
      <td>Montgomery St BART Station (Market St at 2nd St)</td>
      <td>343</td>
      <td>Bryant St at 2nd St</td>
      <td>5034</td>
      <td>2</td>
      <td>1993</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Alternative way
columns_to_check = ['duration_mins', 'station_A_id', 'station_B_id', 'bike_id', 'user_type', 'user_birth_year'] #These were selected based on the unique combination
duplicates_2 = ride_sharing.duplicated(subset = columns_to_check, keep = False)
ride_sharing[duplicates_2]
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
      <th>duration_mins</th>
      <th>station_A_id</th>
      <th>station_A_name</th>
      <th>station_B_id</th>
      <th>station_B_name</th>
      <th>bike_id</th>
      <th>user_type</th>
      <th>user_birth_year</th>
      <th>user_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>566</th>
      <td>9</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>1225</td>
      <td>2</td>
      <td>1993</td>
      <td>0</td>
    </tr>
    <tr>
      <th>604</th>
      <td>9</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>1225</td>
      <td>2</td>
      <td>1993</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5775</th>
      <td>14</td>
      <td>21</td>
      <td>Montgomery St BART Station (Market St at 2nd St)</td>
      <td>52</td>
      <td>McAllister St at Baker St</td>
      <td>5015</td>
      <td>2</td>
      <td>1993</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7610</th>
      <td>10</td>
      <td>30</td>
      <td>San Francisco Caltrain (Townsend St at 4th St)</td>
      <td>6</td>
      <td>The Embarcadero at Sansome St</td>
      <td>4442</td>
      <td>1</td>
      <td>1967</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9696</th>
      <td>17</td>
      <td>22</td>
      <td>Howard St at Beale St</td>
      <td>102</td>
      <td>Irwin St at 8th St</td>
      <td>492</td>
      <td>3</td>
      <td>1961</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9724</th>
      <td>4</td>
      <td>21</td>
      <td>Montgomery St BART Station (Market St at 2nd St)</td>
      <td>343</td>
      <td>Bryant St at 2nd St</td>
      <td>5034</td>
      <td>2</td>
      <td>1993</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11562</th>
      <td>14</td>
      <td>21</td>
      <td>Montgomery St BART Station (Market St at 2nd St)</td>
      <td>52</td>
      <td>McAllister St at Baker St</td>
      <td>5015</td>
      <td>2</td>
      <td>1993</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15217</th>
      <td>17</td>
      <td>22</td>
      <td>Howard St at Beale St</td>
      <td>102</td>
      <td>Irwin St at 8th St</td>
      <td>492</td>
      <td>3</td>
      <td>1961</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18303</th>
      <td>10</td>
      <td>30</td>
      <td>San Francisco Caltrain (Townsend St at 4th St)</td>
      <td>6</td>
      <td>The Embarcadero at Sansome St</td>
      <td>4442</td>
      <td>1</td>
      <td>1967</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18499</th>
      <td>9</td>
      <td>15</td>
      <td>San Francisco Ferry Building (Harry Bridges Pl...</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>4733</td>
      <td>3</td>
      <td>1984</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20170</th>
      <td>4</td>
      <td>21</td>
      <td>Montgomery St BART Station (Market St at 2nd St)</td>
      <td>343</td>
      <td>Bryant St at 2nd St</td>
      <td>5034</td>
      <td>2</td>
      <td>1993</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23293</th>
      <td>9</td>
      <td>15</td>
      <td>San Francisco Ferry Building (Harry Bridges Pl...</td>
      <td>81</td>
      <td>Berry St at 4th St</td>
      <td>4733</td>
      <td>3</td>
      <td>1984</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop duplicates
ride_sharing.drop_duplicates(inplace = True)
ride_sharing.shape


```




    (25756, 9)



## Closing Thoughts
This is the first of 4 data sets I plan to clean. Each one has it's own unique issues. Hopefully, I learn something along the and help a person or two out as well. As most data aficionados will tell you, spending about 80% of your time cleaning and exploring your data is not uncommon. Take your time. Doing it right will not only help you to answer the question(s) at hand, it will help the actual analysis and model performance.

### Don't know where to start?

Try completing these tasks:
- For each DataFrame, inspect the data types of each column and, where needed, clean and convert columns into the correct data type. You should also rename any columns to have more descriptive titles.
- Identify and remove all the duplicate rows in `ride_sharing`.
- Inspect the unique values of all the columns in `airlines` and clean any inconsistencies.
- For the `airlines` DataFrame, create a new column called `International` from `dest_region`, where values representing US regions map to `False` and all other regions map to `True`.
- The `banking` DataFrame contains out of date ages. Update the `Age` column using today's date and the `birth_date` column.
- Clean the `restaurants_new` DataFrame so that it better matches the categories in the `city` and `type` column of the `restaurants` DataFrame. Afterward, given typos in restaurant names, use record linkage to generate possible pairs of rows between `restaurants` and `restaurants_new` using criteria you think is best.

