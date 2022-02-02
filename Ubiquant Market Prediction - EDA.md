## Ubiquant Market Prediction

## 1. Introduction

This competition's objective is predicting value of an obfuscated metric which is relevant for making trading decisions. Dataset contains anonymized features extracted from thousands of investments and an anonymized target. There are 3579 unique investments in training set, but investments in training set doesn't necessarily appear in public or private test set. There are also 1211 unique time IDs in training set. All of the investments doesn't necessarily appear in all time IDs.

There are 303 columns in training and test set after removing row_id. row_id can be removed safely since it is combination of time_id and investment_id columns, and it doesn't contain any additional information. The mentioned columns are:

- `time_id`: Unique ID of the time bucket
- `investment_id`: Unique ID of the investment
- `target`: Anonymized target
- `f0` - `f299`: Anonymized features

In [2]:

```
train_dtypes = {f'f_{i}': np.float32 for i in range(300)}
train_dtypes['investment_id'] = np.uint16
train_dtypes['time_id'] = np.uint16
train_dtypes['target'] = np.float32

df_train = pd.read_csv('../input/ubiquant-market-prediction/train.csv', usecols=list(train_dtypes.keys()), dtype=train_dtypes)
print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
Training Set Shape: (3141410, 303) - Memory Usage: 3619.03 MB
```

As the training set is quite large, it takes more than 8 minutes to read it. Training set can be written as a pickle file so it can be read faster later.

In [3]:

```
df_train.to_pickle('train.pkl')
```

## 2. Evaluation

Submissions are scored on the mean of Pearson correlation coefficient for each time ID. Pearson correlation coefficient (Pearson's r) is a measure of linear correlation between two sets of values. It can be denoted as

r=𝑛∑𝑥𝑦−(∑𝑥)(∑𝑦)[𝑛∑𝑥2−(∑𝑥)2][𝑛∑𝑦2−(∑𝑦)2‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾√]r=n∑xy−(∑x)(∑y)[n∑x2−(∑x)2][n∑y2−(∑y)2]

- 𝑟r = Pearson Correlation Coefficient
- 𝑛n = Number of samples
- 𝑥x = First set of values
- 𝑦y = Second set of values

Mean of Pearson correlation coefficient across time IDs can be denoted as

Mean r=1𝑇∑𝑇𝑖=1𝑡𝑖𝑟Mean r=1T∑i=1Ttir

- 𝑇T = Number of time IDs
- 𝑡𝑖𝑟tir = ith time ID's Pearson correlation coefficient

Fastest way of calculating mean Pearson correlation coefficient is utilizing `groupby` method of `pandas.DataFrame`, however it is not very flexible. The implementations below require predictions column named as *predictions* and they don't output Pearson correlation coefficient of every time ID separately.

In [4]:

```
def pearson_correlation_coefficient(df):
    return df.corr()['target']['predictions']

def mean_pearson_correlation_coefficient(df):
    return np.mean(df[['time_id', 'target', 'predictions']].groupby('time_id').apply(pearson_correlation_coefficient))
```

Pearson correlation coefficient is the ratio between covariances of two sets of values and the product of their standard deviations. That means Pearson correlation coefficient won't change when standard deviations and order of values are kept same for time IDs. Therefore, actual values of predictions doesn't matter and it is similar to a ranking metric in that sense.

In [5]:

```
df_train['predictions'] = np.random.rand(len(df_train))
score = mean_pearson_correlation_coefficient(df_train)
print(f'Pearson correlation coefficient: {score:.6f} - (Predictions mean: {df_train["predictions"].mean():.4f} std: {df_train["predictions"].std():.4f})')
df_train['predictions'] += 999999
score = mean_pearson_correlation_coefficient(df_train)
print(f'Pearson correlation coefficient: {score:.6f} - (Predictions mean: {df_train["predictions"].mean():.4f} std: {df_train["predictions"].std():.4f})')

df_train.drop(columns=['predictions'], inplace=True)
Pearson correlation coefficient: -0.001439 - (Predictions mean: 0.5000 std: 0.2886)
Pearson correlation coefficient: -0.001439 - (Predictions mean: 999999.5000 std: 0.2886)
```

## 3. Target

Target is anonymized and defined as an obfuscated metric relevant for making trading decisions. Target is mean centric and follows a very symmetrical normal distribution which are strong evidences of standardization. Target has very long tails on both ends so distribution truncation or trimming can be quite useful for dealing with outliers.

*unfold_more*Show hidden code

```
target
------
Mean: -0.0211  -  Median: -0.0972  -  Std: 0.9177
Min: -9.4196  -  25%: -0.5005  -  50%: -0.0972  -  75%: 0.3573  -  Max: 12.0386
Skew: 0.9441  -  Kurtosis: 6.2056
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___12_1.png)

## 4. Time IDs

time_id column is the ID code for the time the data was gathered. Time IDs are in chronological order, but the real time gap between time IDs is not constant and will be shorter for the final private test set than in the training set. As mentioned before, there are 1211 unique time IDs in training set and it will be less in private test set.

In [7]:

```
def visualize_time_ids(df, column):
    
    print(f'{column}\n{"-" * len(column)}')
    print(f'Mean: {df[column].mean():.4f}  -  Median: {df[column].median():.4f}  -  Std: {df[column].std():.4f}')
    print(f'Min: {df[column].min():.4f}  -  25%: {df[column].quantile(0.25):.4f}  -  50%: {df[column].quantile(0.5):.4f}  -  75%: {df[column].quantile(0.75):.4f}  -  Max: {df[column].max():.4f}')
    print(f'Skew: {df[column].skew():.4f}  -  Kurtosis: {df[column].kurtosis():.4f}')
    missing_count = df[df[column].isnull()].shape[0]
    total_count = df.shape[0]
    print(f'Missing Values: {missing_count}/{total_count} ({missing_count * 100 / total_count:.4f}%)')

    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100)

    sns.kdeplot(df[column], label=column, fill=True, ax=axes[0])
    axes[0].axvline(df[column].mean(), label='Mean', color='r', linewidth=2, linestyle='--')
    axes[0].axvline(df[column].median(), label='Median', color='b', linewidth=2, linestyle='--')
    axes[0].legend(prop={'size': 15})
    axes[1].plot(df.set_index('time_id')[column], label=column)
    
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5)
        axes[i].tick_params(axis='y', labelsize=12.5)
        axes[i].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('time_id', fontsize=12.5)
    axes[0].set_title(f'{column} Distribution', fontsize=15, pad=12)
    axes[1].set_title(f'{column} as a Function of Time', fontsize=15, pad=12)
    
    plt.show()
```

Number of samples in time IDs are quite different because every investment doesn't necessarily appear in every time ID. Time IDs with less than 2000 samples look like outliers. More than 75% of time IDs have more than 2000 samples. All of the time IDs with less than 2000 samples are observed between time_id 350 and 550. There are couple outliers between time_id 1100 and 1200 as well.

*unfold_more*Show hidden code

```
sample_counts_in_time_ids
-------------------------
Mean: 2594.0628  -  Median: 2489.0000  -  Std: 475.5527
Min: 512.0000  -  25%: 2252.0000  -  50%: 2489.0000  -  75%: 3030.0000  -  Max: 3445.0000
Skew: -0.4783  -  Kurtosis: 1.2167
Missing Values: 0/1211 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___16_1.png)

Target means in time IDs are centered around 0 and they are quite balanced even though there are some outliers, but outliers look very natural. Time IDs with low number of samples match time IDs with high target mean. Target might be correlated with number of samples but it is hard to tell which causes which. Very high and very low target mean values are also observed in the same period (between time_id 350 and 550).

*unfold_more*Show hidden code

```
target_means_in_time_ids
------------------------
Mean: -0.0180  -  Median: -0.0217  -  Std: 0.0911
Min: -0.5614  -  25%: -0.0389  -  50%: -0.0217  -  75%: -0.0080  -  Max: 0.9256
Skew: 4.6483  -  Kurtosis: 42.5545
Missing Values: 0/1211 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___18_1.png)

Target standard deviations in time IDs are centered around 0.9 and they are quite balanced as well, except one of them. One of the time IDs have very low target standard deviation (0.45) and it skews the distribution to left. That outlier doesn't look very natural and it probably is an anomaly. Target standard deviation outliers are also observed in the same period but they are not affected as much as target means and sample counts.

*unfold_more*Show hidden code

```
target_stds_in_time_ids
-----------------------
Mean: 0.9168  -  Median: 0.9216  -  Std: 0.0435
Min: 0.4491  -  25%: 0.8949  -  50%: 0.9216  -  75%: 0.9436  -  Max: 1.2089
Skew: -1.1324  -  Kurtosis: 14.8502
Missing Values: 0/1211 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___20_1.png)

## 5. Investments

investment_id column is the ID code for an investment. There are 3579 unique investments in training set and private test set will include new unseen investments. Investments appear only once in time IDs, so samples are time_id-investment_id combinations. In that case, visualization of sample counts as a function of time, is identical with unique number of investments in time IDs.

In [11]:

```
def visualize_investment_ids(df, column):
    
    print(f'{column}\n{"-" * len(column)}')
    print(f'Mean: {df[column].mean():.4f}  -  Median: {df[column].median():.4f}  -  Std: {df[column].std():.4f}')
    print(f'Min: {df[column].min():.4f}  -  25%: {df[column].quantile(0.25):.4f}  -  50%: {df[column].quantile(0.5):.4f}  -  75%: {df[column].quantile(0.75):.4f}  -  Max: {df[column].max():.4f}')
    print(f'Skew: {df[column].skew():.4f}  -  Kurtosis: {df[column].kurtosis():.4f}')
    missing_count = df[df[column].isnull()].shape[0]
    total_count = df.shape[0]
    print(f'Missing Values: {missing_count}/{total_count} ({missing_count * 100 / total_count:.4f}%)')

    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100)

    sns.kdeplot(df[column], label=column, fill=True, ax=axes[0])
    axes[0].axvline(df[column].mean(), label='Mean', color='r', linewidth=2, linestyle='--')
    axes[0].axvline(df[column].median(), label='Median', color='b', linewidth=2, linestyle='--')
    axes[0].legend(prop={'size': 15})
    axes[1].plot(df.set_index('investment_id')[column], label=column)
    
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5)
        axes[i].tick_params(axis='y', labelsize=12.5)
        axes[i].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('investment_id', fontsize=12.5)
    axes[0].set_title(f'{column} Distribution', fontsize=15, pad=12)
    axes[1].set_title(f'{column} as a Function of Investment', fontsize=15, pad=12)
    
    plt.show()
```

Number of samples per investment is quite different as expected. This is related to the period between time IDs 350 and 550 at when most of the investments are not observed. Number of samples per investment forms a bimodal distribution. Investments with low number of samples are centered around 400 sample count, and investments with high number of samples are centered around 1100 sample count.

*unfold_more*Show hidden code

```
sample_counts_in_investment_ids
-------------------------------
Mean: 877.7340  -  Median: 1009.0000  -  Std: 314.9774
Min: 2.0000  -  25%: 683.0000  -  50%: 1009.0000  -  75%: 1131.0000  -  Max: 1211.0000
Skew: -0.9588  -  Kurtosis: -0.3447
Missing Values: 0/3579 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___24_1.png)

Target means in investments are centered around 0 just like target means in time IDs. It has a very symmetrical and balanced distribution with some small peaks. One investment has a very high target mean (0.795) which could be an outlier. That investment might have higher number of samples between time IDs 350 and 550 because target means are higher at that period.

*unfold_more*Show hidden code

```
target_means_in_investment_ids
------------------------------
Mean: -0.0231  -  Median: -0.0227  -  Std: 0.0464
Min: -0.4830  -  25%: -0.0421  -  50%: -0.0227  -  75%: -0.0017  -  Max: 0.7950
Skew: 0.0005  -  Kurtosis: 41.2269
Missing Values: 0/3579 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___26_1.png)

Target standard deviations in investments are more stable than target means except one investment. That investment has 0 target standard deviation and it has only two samples. That investment's low target standard deviation is related to its low sample count so it might not be an outlier.

In [14]:

```
df = df_train.groupby('investment_id')['target'].std().reset_index().rename(columns={'target': 'target_stds_in_investment_ids'})
visualize_investment_ids(df, 'target_stds_in_investment_ids')
target_stds_in_investment_ids
-----------------------------
Mean: 0.9479  -  Median: 0.9169  -  Std: 0.2147
Min: 0.0000  -  25%: 0.8047  -  50%: 0.9169  -  75%: 1.0586  -  Max: 2.5338
Skew: 1.1224  -  Kurtosis: 3.4262
Missing Values: 0/3579 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___28_1.png)

## 6. Features

There are 300 anonymized continuous features in dataset and they are named from f_0 to f_299. All of the feature distributions, target interactions, feature means and standard deviations along time and investment axis are visualized. There are little summaries of statistical properties displayed before the visualizations.

In [15]:

```
def visualize_feature(df, column):
    
    print(f'{column}\n{"-" * len(column)}')
    print(f'Mean: {df[column].mean():.4f}  -  Median: {df[column].median():.4f}  -  Std: {df[column].std():.4f}')
    print(f'Min: {df[column].min():.4f}  -  25%: {df[column].quantile(0.25):.4f}  -  50%: {df[column].quantile(0.5):.4f}  -  75%: {df[column].quantile(0.75):.4f}  -  Max: {df[column].max():.4f}')
    print(f'Skew: {df[column].skew():.4f}  -  Kurtosis: {df[column].kurtosis():.4f}')
    missing_count = df[df[column].isnull()].shape[0]
    total_count = df.shape[0]
    print(f'Missing Values: {missing_count}/{total_count} ({missing_count * 100 / total_count:.4f}%)')

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(24, 22), dpi=100)

    sns.kdeplot(df[column], label=column, fill=True, ax=axes[0][0])
    axes[0][0].axvline(df[column].mean(), label='Mean', color='r', linewidth=2, linestyle='--')
    axes[0][0].axvline(df[column].median(), label='Median', color='b', linewidth=2, linestyle='--')
    axes[0][0].legend(prop={'size': 15})
    sns.scatterplot(x=df[column], y=df['target'], ax=axes[0][1])
    
    df_feature_means_in_time_ids = df_train.groupby('time_id')[column].mean().reset_index().rename(columns={column: f'{column}_means_in_time_ids'})
    axes[1][0].plot(df_feature_means_in_time_ids.set_index('time_id')[f'{column}_means_in_time_ids'], label=f'{column}_means_in_time_ids')
    df_feature_stds_in_time_ids = df_train.groupby('time_id')[column].std().reset_index().rename(columns={column: f'{column}_stds_in_time_ids'})
    axes[1][1].plot(df_feature_stds_in_time_ids.set_index('time_id')[f'{column}_stds_in_time_ids'], label=f'{column}_stds_in_time_ids')
    
    df_feature_means_in_investment_ids = df_train.groupby('investment_id')[column].mean().reset_index().rename(columns={column: f'{column}_means_in_investment_ids'})
    axes[2][0].plot(df_feature_means_in_investment_ids.set_index('investment_id')[f'{column}_means_in_investment_ids'], label=f'{column}_means_in_investment_ids')
    df_feature_stds_in_investment_ids = df_train.groupby('investment_id')[column].std().reset_index().rename(columns={column: f'{column}_stds_in_investment_ids'})
    axes[2][1].plot(df_feature_stds_in_investment_ids.set_index('investment_id')[f'{column}_stds_in_investment_ids'], label=f'{column}_stds_in_investment_ids')

    for i in range(3):
        for j in range(2):
            axes[i][j].tick_params(axis='x', labelsize=12.5)
            axes[i][j].tick_params(axis='y', labelsize=12.5)
            axes[i][j].set_ylabel('')
            
    axes[0][0].set_xlabel('')
    axes[0][1].set_xlabel(column, fontsize=12.5)
    axes[0][1].set_ylabel('target', fontsize=12.5)
    
    for i in range(2):
        axes[1][i].set_xlabel('time_id', fontsize=12.5)
        axes[1][i].set_ylabel(column, fontsize=12.5)
        
    for i in range(2):
        axes[2][i].set_xlabel('investment_id', fontsize=12.5)
        axes[2][i].set_ylabel(column, fontsize=12.5)
        
    axes[0][0].set_title(f'{column} Distribution', fontsize=15, pad=12)
    axes[0][1].set_title(f'{column} vs Target', fontsize=15, pad=12)
    axes[1][0].set_title(f'{column} Means as a Function of Time', fontsize=15, pad=12)
    axes[1][1].set_title(f'{column} Stds as a Function of Time', fontsize=15, pad=12)
    axes[2][0].set_title(f'{column} Means as a Function of Investment', fontsize=15, pad=12)
    axes[2][1].set_title(f'{column} Stds as a Function of Investment', fontsize=15, pad=12)
    
    plt.show()
```

All of the features are zero-centered and they have standard deviation of one since they are standardized during the anonymization process. Most of the features have symmetrical normal distributions but some of them have very extreme outliers which are skewing their distributions.

Feature means and standard deviations vary between different time IDs and investments. It looks like feature means and standard deviations are dependent to time. They make sharp transitions on some periods. Feature standard deviations are more likely to make sharp transitions on different periods however feature mean outliers are observed in the same period most of the time. Feature means and standard deviations per investment looks randomly distributed among investments because it is related to those investment's time IDs.

*unfold_more*Show hidden code

```
f_0
---
Mean: 0.0016  -  Median: 0.2431  -  Std: 1.0797
Min: -17.6579  -  25%: -0.4071  -  50%: 0.2431  -  75%: 0.6650  -  Max: 7.8453
Skew: -1.7729  -  Kurtosis: 6.2544
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_1.png)

```
f_1
---
Mean: -0.0126  -  Median: -0.0487  -  Std: 1.0310
Min: -6.5795  -  25%: -0.6814  -  50%: -0.0487  -  75%: 0.6087  -  Max: 8.7072
Skew: 0.3430  -  Kurtosis: 1.1923
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_3.png)

```
f_2
---
Mean: 0.0085  -  Median: 0.0435  -  Std: 1.0303
Min: -8.6443  -  25%: -0.6531  -  50%: 0.0435  -  75%: 0.6587  -  Max: 8.0093
Skew: -0.1281  -  Kurtosis: 0.6100
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_5.png)

```
f_3
---
Mean: -0.0200  -  Median: -0.2640  -  Std: 0.9603
Min: -18.0043  -  25%: -0.4496  -  50%: -0.2640  -  75%: 0.0776  -  Max: 47.0633
Skew: 5.6991  -  Kurtosis: 79.4873
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_7.png)

```
f_4
---
Mean: -0.0054  -  Median: -0.1909  -  Std: 0.9895
Min: -4.0000  -  25%: -0.3532  -  50%: -0.1909  -  75%: 0.0278  -  Max: 76.6287
Skew: 20.9345  -  Kurtosis: 897.4894
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_9.png)

```
f_5
---
Mean: -0.0030  -  Median: -0.0214  -  Std: 1.1046
Min: -8.8337  -  25%: -0.7071  -  50%: -0.0214  -  75%: 0.6836  -  Max: 7.6462
Skew: 0.0708  -  Kurtosis: 0.4010
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_11.png)

```
f_6
---
Mean: 0.0019  -  Median: 0.0144  -  Std: 1.0678
Min: -6.3843  -  25%: -0.7911  -  50%: 0.0144  -  75%: 0.8008  -  Max: 6.7781
Skew: -0.0386  -  Kurtosis: -0.3178
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_13.png)

```
f_7
---
Mean: 0.0059  -  Median: -0.0588  -  Std: 1.0046
Min: -13.9481  -  25%: -0.6300  -  50%: -0.0588  -  75%: 0.6883  -  Max: 5.7220
Skew: -0.2503  -  Kurtosis: 2.7834
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_15.png)

```
f_8
---
Mean: 0.0128  -  Median: 0.1590  -  Std: 1.0736
Min: -33.4492  -  25%: 0.1064  -  50%: 0.1590  -  75%: 0.2333  -  Max: 15.4989
Skew: -5.4106  -  Kurtosis: 45.7325
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_17.png)

```
f_9
---
Mean: -0.0032  -  Median: 0.0151  -  Std: 1.1454
Min: -7.7821  -  25%: -0.8624  -  50%: 0.0151  -  75%: 0.8511  -  Max: 7.3205
Skew: -0.0309  -  Kurtosis: -0.2675
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_19.png)

```
f_10
----
Mean: 0.0094  -  Median: 0.1342  -  Std: 0.9876
Min: -75.9329  -  25%: -0.0430  -  50%: 0.1342  -  75%: 0.3064  -  Max: 26.5775
Skew: -15.7753  -  Kurtosis: 483.6620
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_21.png)

```
f_11
----
Mean: 0.0104  -  Median: 0.0827  -  Std: 1.0781
Min: -32.9261  -  25%: -0.3809  -  50%: 0.0827  -  75%: 0.5129  -  Max: 32.0339
Skew: -2.1746  -  Kurtosis: 23.9375
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_23.png)

```
f_12
----
Mean: 0.0016  -  Median: 0.1829  -  Std: 1.0925
Min: -50.2530  -  25%: -0.5055  -  50%: 0.1829  -  75%: 0.7280  -  Max: 7.3763
Skew: -2.1538  -  Kurtosis: 20.1334
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_25.png)

```
f_13
----
Mean: -0.0039  -  Median: -0.0101  -  Std: 1.0777
Min: -77.1615  -  25%: -0.2125  -  50%: -0.0101  -  75%: 0.2008  -  Max: 66.4168
Skew: -0.2725  -  Kurtosis: 169.3031
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_27.png)

```
f_14
----
Mean: -0.0027  -  Median: 0.0475  -  Std: 1.1392
Min: -27.9702  -  25%: -0.7572  -  50%: 0.0475  -  75%: 0.7495  -  Max: 12.4072
Skew: -0.1801  -  Kurtosis: 2.4420
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_29.png)

```
f_15
----
Mean: -0.0110  -  Median: -0.2467  -  Std: 1.0474
Min: -5.8133  -  25%: -0.7284  -  50%: -0.2467  -  75%: 0.4772  -  Max: 12.5798
Skew: 1.4218  -  Kurtosis: 3.5668
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_31.png)

```
f_16
----
Mean: -0.0032  -  Median: -0.2574  -  Std: 1.0475
Min: -4.6685  -  25%: -0.8133  -  50%: -0.2574  -  75%: 0.7147  -  Max: 6.8552
Skew: 0.7069  -  Kurtosis: -0.0742
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_33.png)

```
f_17
----
Mean: 0.0072  -  Median: -0.1788  -  Std: 1.0254
Min: -14.2966  -  25%: -0.7351  -  50%: -0.1788  -  75%: 0.5572  -  Max: 22.3917
Skew: 1.1236  -  Kurtosis: 2.3765
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_35.png)

```
f_18
----
Mean: 0.0011  -  Median: -0.0166  -  Std: 1.1430
Min: -22.8167  -  25%: -0.7879  -  50%: -0.0166  -  75%: 0.8012  -  Max: 27.8371
Skew: 0.0286  -  Kurtosis: 1.0525
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_37.png)

```
f_19
----
Mean: -0.0092  -  Median: -0.0116  -  Std: 1.0665
Min: -14.2361  -  25%: -0.5016  -  50%: -0.0116  -  75%: 0.4793  -  Max: 14.8423
Skew: -0.0457  -  Kurtosis: 6.9210
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_39.png)

```
f_20
----
Mean: 0.0020  -  Median: 0.0258  -  Std: 1.0291
Min: -5.9999  -  25%: -0.6710  -  50%: 0.0258  -  75%: 0.7221  -  Max: 5.5362
Skew: -0.2328  -  Kurtosis: 0.0649
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_41.png)

```
f_21
----
Mean: -0.0078  -  Median: -0.0551  -  Std: 1.0979
Min: -12.9319  -  25%: -0.6584  -  50%: -0.0551  -  75%: 0.6184  -  Max: 13.9924
Skew: 0.1622  -  Kurtosis: 2.2211
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_43.png)

```
f_22
----
Mean: 0.0153  -  Median: 0.3465  -  Std: 1.0027
Min: -6.4482  -  25%: -0.0921  -  50%: 0.3465  -  75%: 0.6199  -  Max: 4.3041
Skew: -1.4562  -  Kurtosis: 1.5235
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_45.png)

```
f_23
----
Mean: -0.0055  -  Median: -0.2226  -  Std: 1.0243
Min: -16.8394  -  25%: -0.6186  -  50%: -0.2226  -  75%: 0.3762  -  Max: 63.5101
Skew: 4.6494  -  Kurtosis: 114.8454
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_47.png)

```
f_24
----
Mean: -0.0032  -  Median: -0.0042  -  Std: 1.0187
Min: -17.3276  -  25%: -0.6465  -  50%: -0.0042  -  75%: 0.6577  -  Max: 6.1145
Skew: -0.6262  -  Kurtosis: 6.0167
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_49.png)

```
f_25
----
Mean: -0.0150  -  Median: -0.2918  -  Std: 1.0122
Min: -5.8694  -  25%: -0.6886  -  50%: -0.2918  -  75%: 0.3799  -  Max: 13.3130
Skew: 1.9084  -  Kurtosis: 5.9109
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_51.png)

```
f_26
----
Mean: 0.0098  -  Median: 0.2933  -  Std: 1.0716
Min: -15.5388  -  25%: -0.4425  -  50%: 0.2933  -  75%: 0.6507  -  Max: 8.8042
Skew: -1.5090  -  Kurtosis: 4.8635
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_53.png)

```
f_27
----
Mean: 0.0056  -  Median: 0.0665  -  Std: 1.1369
Min: -7.6854  -  25%: -0.8217  -  50%: 0.0665  -  75%: 0.8240  -  Max: 23.1031
Skew: -0.0142  -  Kurtosis: 0.7818
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_55.png)

```
f_28
----
Mean: -0.0118  -  Median: -0.0146  -  Std: 1.0318
Min: -4.7613  -  25%: -0.8643  -  50%: -0.0146  -  75%: 0.8430  -  Max: 4.4403
Skew: -0.0005  -  Kurtosis: -0.9450
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_57.png)

```
f_29
----
Mean: 0.0124  -  Median: 0.1093  -  Std: 1.0611
Min: -12.8370  -  25%: -0.5922  -  50%: 0.1093  -  75%: 0.6964  -  Max: 22.4264
Skew: -0.1779  -  Kurtosis: 4.5356
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_59.png)

```
f_30
----
Mean: 0.0038  -  Median: 0.2187  -  Std: 1.0585
Min: -20.3543  -  25%: -0.2374  -  50%: 0.2187  -  75%: 0.5208  -  Max: 10.2690
Skew: -2.6315  -  Kurtosis: 14.1928
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_61.png)

```
f_31
----
Mean: -0.0122  -  Median: -0.2571  -  Std: 1.0379
Min: -6.3424  -  25%: -0.7384  -  50%: -0.2571  -  75%: 0.4674  -  Max: 20.5845
Skew: 1.5854  -  Kurtosis: 4.9278
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_63.png)

```
f_32
----
Mean: 0.0003  -  Median: -0.0595  -  Std: 1.0688
Min: -10.8363  -  25%: -0.6262  -  50%: -0.0595  -  75%: 0.5264  -  Max: 19.6412
Skew: 1.1095  -  Kurtosis: 6.4720
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_65.png)

```
f_33
----
Mean: 0.0109  -  Median: 0.2513  -  Std: 1.0084
Min: -25.0083  -  25%: 0.1172  -  50%: 0.2513  -  75%: 0.3215  -  Max: 12.0810
Skew: -6.2053  -  Kurtosis: 54.3302
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_67.png)

```
f_34
----
Mean: 0.0017  -  Median: -0.0049  -  Std: 1.1046
Min: -7.6849  -  25%: -0.6742  -  50%: -0.0049  -  75%: 0.7211  -  Max: 6.9498
Skew: -0.1054  -  Kurtosis: 0.4511
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_69.png)

```
f_35
----
Mean: -0.0071  -  Median: -0.0577  -  Std: 1.0137
Min: -4.7776  -  25%: -0.6779  -  50%: -0.0577  -  75%: 0.5840  -  Max: 9.1610
Skew: 0.7581  -  Kurtosis: 2.7406
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_71.png)

```
f_36
----
Mean: -0.0093  -  Median: -0.1966  -  Std: 1.0071
Min: -4.8372  -  25%: -0.7177  -  50%: -0.1966  -  75%: 0.5198  -  Max: 8.5732
Skew: 1.0569  -  Kurtosis: 1.9716
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_73.png)

```
f_37
----
Mean: 0.0023  -  Median: 0.1737  -  Std: 1.0547
Min: -43.6004  -  25%: -0.4520  -  50%: 0.1737  -  75%: 0.6537  -  Max: 12.6634
Skew: -2.7269  -  Kurtosis: 26.4433
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_75.png)

```
f_38
----
Mean: -0.0017  -  Median: -0.2409  -  Std: 1.0334
Min: -14.6775  -  25%: -0.3731  -  50%: -0.2409  -  75%: 0.0025  -  Max: 56.9502
Skew: 10.5531  -  Kurtosis: 230.1270
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_77.png)

```
f_39
----
Mean: -0.0080  -  Median: -0.2915  -  Std: 1.0052
Min: -10.7280  -  25%: -0.4953  -  50%: -0.2915  -  75%: 0.1073  -  Max: 42.7226
Skew: 6.3353  -  Kurtosis: 89.2580
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_79.png)

```
f_40
----
Mean: -0.0060  -  Median: -0.0228  -  Std: 1.0582
Min: -60.6195  -  25%: -0.4360  -  50%: -0.0228  -  75%: 0.4020  -  Max: 63.6665
Skew: -0.0361  -  Kurtosis: 154.1865
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_81.png)

```
f_41
----
Mean: 0.4064  -  Median: 0.0000  -  Std: 0.9312
Min: -22.9187  -  25%: 0.0000  -  50%: 0.0000  -  75%: 1.1940  -  Max: 6.9331
Skew: -0.4061  -  Kurtosis: 3.1515
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_83.png)

```
f_42
----
Mean: -0.0199  -  Median: -0.2136  -  Std: 1.0249
Min: -12.5488  -  25%: -0.5626  -  50%: -0.2136  -  75%: 0.2870  -  Max: 24.2626
Skew: 2.6877  -  Kurtosis: 18.5519
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_85.png)

```
f_43
----
Mean: -0.0053  -  Median: -0.0112  -  Std: 1.1003
Min: -6.5681  -  25%: -0.8714  -  50%: -0.0112  -  75%: 0.8603  -  Max: 6.9272
Skew: 0.0098  -  Kurtosis: -0.5683
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_87.png)

```
f_44
----
Mean: -0.0129  -  Median: -0.0408  -  Std: 1.0888
Min: -10.0436  -  25%: -0.7058  -  50%: -0.0408  -  75%: 0.6138  -  Max: 37.0656
Skew: 0.3925  -  Kurtosis: 2.7965
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_89.png)

```
f_45
----
Mean: -0.0122  -  Median: -0.2429  -  Std: 1.0225
Min: -6.7291  -  25%: -0.6145  -  50%: -0.2429  -  75%: 0.3428  -  Max: 50.3262
Skew: 3.5767  -  Kurtosis: 44.0046
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_91.png)

```
f_46
----
Mean: -0.0011  -  Median: 0.0220  -  Std: 1.0344
Min: -5.9348  -  25%: -0.8078  -  50%: 0.0220  -  75%: 0.8308  -  Max: 5.5031
Skew: -0.1302  -  Kurtosis: -0.5667
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_93.png)

```
f_47
----
Mean: -0.0163  -  Median: -0.2269  -  Std: 1.0329
Min: -7.8835  -  25%: -0.6802  -  50%: -0.2269  -  75%: 0.4083  -  Max: 24.5384
Skew: 2.2443  -  Kurtosis: 13.9863
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_95.png)

```
f_48
----
Mean: -0.0043  -  Median: -0.5129  -  Std: 1.0815
Min: -6.3733  -  25%: -0.8630  -  50%: -0.5129  -  75%: 1.1169  -  Max: 5.6139
Skew: 0.2233  -  Kurtosis: -1.2417
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_97.png)

```
f_49
----
Mean: -0.0050  -  Median: 0.0839  -  Std: 1.0219
Min: -72.2499  -  25%: -0.5610  -  50%: 0.0839  -  75%: 0.6400  -  Max: 48.6948
Skew: -2.0272  -  Kurtosis: 102.1468
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_99.png)

```
f_50
----
Mean: -0.0081  -  Median: -0.0068  -  Std: 1.0059
Min: -3.7265  -  25%: -0.8636  -  50%: -0.0068  -  75%: 0.8486  -  Max: 3.2836
Skew: -0.0038  -  Kurtosis: -1.1065
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_101.png)

```
f_51
----
Mean: -0.0110  -  Median: -0.0187  -  Std: 1.0091
Min: -3.6186  -  25%: -0.8696  -  50%: -0.0187  -  75%: 0.8415  -  Max: 4.3455
Skew: 0.0244  -  Kurtosis: -1.0897
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_103.png)

```
f_52
----
Mean: -0.0045  -  Median: -0.0508  -  Std: 1.0670
Min: -25.2021  -  25%: -0.4925  -  50%: -0.0508  -  75%: 0.4259  -  Max: 18.7376
Skew: -0.2041  -  Kurtosis: 16.2912
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_105.png)

```
f_53
----
Mean: 0.0012  -  Median: -0.1375  -  Std: 1.0820
Min: -14.7198  -  25%: -0.5940  -  50%: -0.1375  -  75%: 0.4265  -  Max: 16.8608
Skew: 1.1089  -  Kurtosis: 6.0960
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_107.png)

```
f_54
----
Mean: -0.0148  -  Median: -0.1932  -  Std: 0.9886
Min: -6.0994  -  25%: -0.6549  -  50%: -0.1932  -  75%: 0.4267  -  Max: 52.1896
Skew: 1.8952  -  Kurtosis: 22.8977
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_109.png)

```
f_55
----
Mean: -0.0080  -  Median: -0.3295  -  Std: 1.0920
Min: -10.8234  -  25%: -0.4635  -  50%: -0.3295  -  75%: 0.1464  -  Max: 45.6681
Skew: 4.6706  -  Kurtosis: 56.0544
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_111.png)

```
f_56
----
Mean: 0.0059  -  Median: 0.3582  -  Std: 1.0533
Min: -11.2591  -  25%: 0.1172  -  50%: 0.3582  -  75%: 0.5024  -  Max: 7.1186
Skew: -1.9547  -  Kurtosis: 4.3212
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_113.png)

```
f_57
----
Mean: -0.0018  -  Median: -0.0054  -  Std: 1.1045
Min: -12.5389  -  25%: -0.6022  -  50%: -0.0054  -  75%: 0.5874  -  Max: 10.9299
Skew: -0.0165  -  Kurtosis: 2.9174
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_115.png)

```
f_58
----
Mean: -0.0097  -  Median: -0.0013  -  Std: 1.0016
Min: -11.2684  -  25%: -0.6197  -  50%: -0.0013  -  75%: 0.5003  -  Max: 13.8653
Skew: 0.9718  -  Kurtosis: 5.0634
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_117.png)

```
f_59
----
Mean: -0.0200  -  Median: -0.2759  -  Std: 0.9783
Min: -6.4016  -  25%: -0.5859  -  50%: -0.2759  -  75%: 0.2542  -  Max: 28.1815
Skew: 3.8040  -  Kurtosis: 31.7468
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_119.png)

```
f_60
----
Mean: 0.0071  -  Median: 0.2858  -  Std: 1.0188
Min: -11.0510  -  25%: -0.5389  -  50%: 0.2858  -  75%: 0.6223  -  Max: 6.9785
Skew: -1.3232  -  Kurtosis: 3.9900
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_121.png)

```
f_61
----
Mean: 0.0097  -  Median: -0.1169  -  Std: 1.0090
Min: -2.9553  -  25%: -0.5786  -  50%: -0.1169  -  75%: 0.4018  -  Max: 14.8067
Skew: 2.1844  -  Kurtosis: 11.8430
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_123.png)

```
f_62
----
Mean: -0.0113  -  Median: -0.1652  -  Std: 1.0064
Min: -11.2667  -  25%: -0.1949  -  50%: -0.1652  -  75%: -0.1342  -  Max: 37.1164
Skew: 7.0891  -  Kurtosis: 64.4766
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_125.png)

```
f_63
----
Mean: -0.0086  -  Median: -0.2133  -  Std: 1.0439
Min: -7.4350  -  25%: -0.3594  -  50%: -0.2133  -  75%: -0.1227  -  Max: 16.4609
Skew: 3.9187  -  Kurtosis: 21.9670
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_127.png)

```
f_64
----
Mean: -0.0079  -  Median: -0.2629  -  Std: 1.0092
Min: -6.7977  -  25%: -0.5581  -  50%: -0.2629  -  75%: 0.2252  -  Max: 56.8316
Skew: 6.8444  -  Kurtosis: 142.5033
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_129.png)

```
f_65
----
Mean: -0.0227  -  Median: -0.1883  -  Std: 1.0390
Min: -5.8942  -  25%: -0.7981  -  50%: -0.1883  -  75%: 0.5922  -  Max: 17.7554
Skew: 0.9390  -  Kurtosis: 2.0195
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_131.png)

```
f_66
----
Mean: -0.0016  -  Median: -0.5059  -  Std: 1.0454
Min: -5.2732  -  25%: -0.5678  -  50%: -0.5059  -  75%: 0.0669  -  Max: 5.9344
Skew: 1.3090  -  Kurtosis: 0.3815
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_133.png)

```
f_67
----
Mean: -0.0001  -  Median: 0.0281  -  Std: 1.0413
Min: -16.9569  -  25%: -0.5351  -  50%: 0.0281  -  75%: 0.5646  -  Max: 10.2805
Skew: -0.6605  -  Kurtosis: 7.3346
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_135.png)

```
f_68
----
Mean: -0.0080  -  Median: -0.2833  -  Std: 1.0327
Min: -7.9430  -  25%: -0.6194  -  50%: -0.2833  -  75%: 0.3082  -  Max: 20.7874
Skew: 2.2826  -  Kurtosis: 9.5693
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_137.png)

```
f_69
----
Mean: -0.0053  -  Median: -0.1014  -  Std: 1.1216
Min: -10.8554  -  25%: -0.6462  -  50%: -0.1014  -  75%: 0.5306  -  Max: 13.0025
Skew: 0.6769  -  Kurtosis: 3.3132
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_139.png)

```
f_70
----
Mean: -0.0033  -  Median: -0.3019  -  Std: 1.0251
Min: -5.2706  -  25%: -0.7674  -  50%: -0.3019  -  75%: 0.5217  -  Max: 7.3480
Skew: 1.1706  -  Kurtosis: 1.1887
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_141.png)

```
f_71
----
Mean: -0.0165  -  Median: -0.2997  -  Std: 0.9888
Min: -9.3585  -  25%: -0.5503  -  50%: -0.2997  -  75%: 0.1921  -  Max: 41.6249
Skew: 4.3961  -  Kurtosis: 42.3732
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_143.png)

```
f_72
----
Mean: 0.0129  -  Median: 0.0104  -  Std: 1.0156
Min: -3.7243  -  25%: -0.8462  -  50%: 0.0104  -  75%: 0.8712  -  Max: 4.1717
Skew: 0.0056  -  Kurtosis: -1.0773
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_145.png)

```
f_73
----
Mean: -0.0046  -  Median: -0.1548  -  Std: 1.0004
Min: -4.3558  -  25%: -0.6881  -  50%: -0.1548  -  75%: 0.5027  -  Max: 12.0415
Skew: 0.9447  -  Kurtosis: 1.8590
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_147.png)

```
f_74
----
Mean: 0.0132  -  Median: 0.0545  -  Std: 0.9884
Min: -67.4064  -  25%: 0.0352  -  50%: 0.0545  -  75%: 0.1135  -  Max: 22.3334
Skew: -23.9494  -  Kurtosis: 853.0006
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_149.png)

```
f_75
----
Mean: -0.0067  -  Median: -0.0303  -  Std: 1.0319
Min: -9.1962  -  25%: -0.6599  -  50%: -0.0303  -  75%: 0.6216  -  Max: 12.3291
Skew: 0.7250  -  Kurtosis: 4.0201
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_151.png)

```
f_76
----
Mean: 0.0074  -  Median: 0.1143  -  Std: 1.0901
Min: -10.7591  -  25%: -0.5695  -  50%: 0.1143  -  75%: 0.6934  -  Max: 11.9219
Skew: -0.6235  -  Kurtosis: 2.3017
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_153.png)

```
f_77
----
Mean: -0.0155  -  Median: -0.2218  -  Std: 1.0071
Min: -11.4655  -  25%: -0.4759  -  50%: -0.2218  -  75%: 0.1575  -  Max: 39.1426
Skew: 6.1999  -  Kurtosis: 78.3051
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_155.png)

```
f_78
----
Mean: -0.0146  -  Median: -0.0535  -  Std: 0.9568
Min: -48.2367  -  25%: -0.1807  -  50%: -0.0535  -  75%: 0.0960  -  Max: 68.4235
Skew: 13.8614  -  Kurtosis: 547.7734
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_157.png)

```
f_79
----
Mean: -0.0123  -  Median: -0.2614  -  Std: 1.0322
Min: -7.2594  -  25%: -0.6931  -  50%: -0.2614  -  75%: 0.3964  -  Max: 15.6351
Skew: 1.7684  -  Kurtosis: 5.5096
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_159.png)

```
f_80
----
Mean: -0.0198  -  Median: -0.2298  -  Std: 1.0303
Min: -12.3016  -  25%: -0.5549  -  50%: -0.2298  -  75%: 0.2483  -  Max: 49.9506
Skew: 5.8405  -  Kurtosis: 99.8829
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_161.png)

```
f_81
----
Mean: 0.0032  -  Median: 0.0645  -  Std: 1.1223
Min: -8.6438  -  25%: -0.8188  -  50%: 0.0645  -  75%: 0.7989  -  Max: 8.9692
Skew: -0.1187  -  Kurtosis: 0.2350
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_163.png)

```
f_82
----
Mean: -0.0012  -  Median: 0.0295  -  Std: 1.0315
Min: -5.1430  -  25%: -0.7897  -  50%: 0.0295  -  75%: 0.8264  -  Max: 5.5522
Skew: -0.0893  -  Kurtosis: -0.5590
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_165.png)

```
f_83
----
Mean: -0.0199  -  Median: -0.0972  -  Std: 0.9758
Min: -14.0218  -  25%: -0.4668  -  50%: -0.0972  -  75%: 0.3219  -  Max: 65.4061
Skew: 9.0431  -  Kurtosis: 256.6258
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_167.png)

```
f_84
----
Mean: 0.0039  -  Median: 0.1276  -  Std: 1.0411
Min: -59.1720  -  25%: -0.3569  -  50%: 0.1276  -  75%: 0.5468  -  Max: 68.7287
Skew: -0.1753  -  Kurtosis: 162.9625
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_169.png)

```
f_85
----
Mean: -0.0016  -  Median: 0.0041  -  Std: 1.0388
Min: -5.7214  -  25%: -0.8437  -  50%: 0.0041  -  75%: 0.8518  -  Max: 5.2388
Skew: -0.0320  -  Kurtosis: -0.8491
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_171.png)

```
f_86
----
Mean: -0.0127  -  Median: -0.0151  -  Std: 1.0202
Min: -4.5569  -  25%: -0.8688  -  50%: -0.0151  -  75%: 0.8406  -  Max: 4.0206
Skew: 0.0094  -  Kurtosis: -1.0234
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_173.png)

```
f_87
----
Mean: -0.0056  -  Median: -0.2942  -  Std: 1.0487
Min: -16.6433  -  25%: -0.5431  -  50%: -0.2942  -  75%: 0.1722  -  Max: 45.1489
Skew: 4.6183  -  Kurtosis: 45.8040
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_175.png)

```
f_88
----
Mean: 0.0093  -  Median: 0.1035  -  Std: 1.0432
Min: -8.9393  -  25%: -0.6868  -  50%: 0.1035  -  75%: 0.7231  -  Max: 13.9898
Skew: -0.2242  -  Kurtosis: 1.4049
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_177.png)

```
f_89
----
Mean: -0.0135  -  Median: -0.0310  -  Std: 0.9611
Min: -18.2109  -  25%: -0.5058  -  50%: -0.0310  -  75%: 0.4647  -  Max: 33.2291
Skew: 0.2723  -  Kurtosis: 17.3193
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_179.png)

```
f_90
----
Mean: -0.0034  -  Median: -0.1066  -  Std: 1.1062
Min: -6.9771  -  25%: -0.7576  -  50%: -0.1066  -  75%: 0.6873  -  Max: 12.6222
Skew: 0.4570  -  Kurtosis: 0.8171
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_181.png)

```
f_91
----
Mean: -0.0107  -  Median: -0.1701  -  Std: 1.0131
Min: -8.0282  -  25%: -0.5434  -  50%: -0.1701  -  75%: 0.2727  -  Max: 10.7286
Skew: 1.6643  -  Kurtosis: 5.1073
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_183.png)

```
f_92
----
Mean: -0.0079  -  Median: -0.2182  -  Std: 1.0207
Min: -8.6550  -  25%: -0.6182  -  50%: -0.2182  -  75%: 0.3399  -  Max: 21.2228
Skew: 2.8631  -  Kurtosis: 17.5751
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_185.png)

```
f_93
----
Mean: 0.0067  -  Median: 0.1527  -  Std: 1.1033
Min: -9.4567  -  25%: -0.6585  -  50%: 0.1527  -  75%: 0.7492  -  Max: 8.1056
Skew: -0.5725  -  Kurtosis: 0.9611
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_187.png)

```
f_94
----
Mean: -0.0007  -  Median: 0.0014  -  Std: 1.1254
Min: -14.2393  -  25%: -0.7687  -  50%: 0.0014  -  75%: 0.7507  -  Max: 10.0906
Skew: -0.0242  -  Kurtosis: 0.6328
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_189.png)

```
f_95
----
Mean: 0.0097  -  Median: -0.0167  -  Std: 1.0049
Min: -14.5071  -  25%: -0.6286  -  50%: -0.0167  -  75%: 0.6466  -  Max: 56.5416
Skew: 1.6862  -  Kurtosis: 85.6856
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_191.png)

```
f_96
----
Mean: -0.0079  -  Median: -0.0738  -  Std: 1.0864
Min: -10.4595  -  25%: -0.7276  -  50%: -0.0738  -  75%: 0.6920  -  Max: 9.5182
Skew: 0.1782  -  Kurtosis: 0.7851
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_193.png)

```
f_97
----
Mean: -0.0090  -  Median: -0.2637  -  Std: 0.9956
Min: -2.8244  -  25%: -0.6549  -  50%: -0.2637  -  75%: 0.3466  -  Max: 18.6149
Skew: 2.3652  -  Kurtosis: 10.8635
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_195.png)

```
f_98
----
Mean: -0.0033  -  Median: 0.0773  -  Std: 1.0181
Min: -6.7147  -  25%: -0.6551  -  50%: 0.0773  -  75%: 0.7824  -  Max: 6.6889
Skew: -0.5948  -  Kurtosis: 0.4458
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_197.png)

```
f_99
----
Mean: -0.0026  -  Median: -0.2106  -  Std: 1.0418
Min: -11.1618  -  25%: -0.3282  -  50%: -0.2106  -  75%: -0.1110  -  Max: 45.7469
Skew: 5.9985  -  Kurtosis: 67.4314
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_199.png)

```
f_100
-----
Mean: -0.0129  -  Median: -0.2627  -  Std: 1.0788
Min: -6.5046  -  25%: -0.6952  -  50%: -0.2627  -  75%: 0.5854  -  Max: 17.3083
Skew: 0.9368  -  Kurtosis: 2.3487
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_201.png)

```
f_101
-----
Mean: -0.0083  -  Median: -0.2227  -  Std: 1.0199
Min: -16.0227  -  25%: -0.5592  -  50%: -0.2227  -  75%: 0.2341  -  Max: 54.5304
Skew: 6.5641  -  Kurtosis: 143.3144
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_203.png)

```
f_102
-----
Mean: -0.0034  -  Median: 0.2612  -  Std: 1.1217
Min: -56.3205  -  25%: -0.7578  -  50%: 0.2612  -  75%: 0.7153  -  Max: 10.4215
Skew: -1.8885  -  Kurtosis: 29.2480
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_205.png)

```
f_103
-----
Mean: 0.0022  -  Median: -0.0255  -  Std: 1.0924
Min: -9.4289  -  25%: -0.6706  -  50%: -0.0255  -  75%: 0.6609  -  Max: 9.2796
Skew: 0.0933  -  Kurtosis: 1.1643
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_207.png)

```
f_104
-----
Mean: -0.0022  -  Median: 0.0113  -  Std: 1.1374
Min: -27.9702  -  25%: -0.7480  -  50%: 0.0113  -  75%: 0.7560  -  Max: 58.8133
Skew: -0.0834  -  Kurtosis: 4.5993
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_209.png)

```
f_105
-----
Mean: -0.0015  -  Median: 0.0047  -  Std: 1.1105
Min: -9.1003  -  25%: -0.7376  -  50%: 0.0047  -  75%: 0.7285  -  Max: 8.5740
Skew: -0.0009  -  Kurtosis: 0.7124
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_211.png)

```
f_106
-----
Mean: -0.0054  -  Median: -0.2187  -  Std: 1.0144
Min: -5.4885  -  25%: -0.7044  -  50%: -0.2187  -  75%: 0.4603  -  Max: 14.9097
Skew: 1.4956  -  Kurtosis: 4.0743
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_213.png)

```
f_107
-----
Mean: -0.0025  -  Median: 0.0042  -  Std: 1.1329
Min: -12.2047  -  25%: -0.7629  -  50%: 0.0042  -  75%: 0.7527  -  Max: 15.2865
Skew: 0.0843  -  Kurtosis: 1.4795
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_215.png)

```
f_108
-----
Mean: 0.0221  -  Median: 0.0817  -  Std: 0.9398
Min: -34.9519  -  25%: 0.0621  -  50%: 0.0817  -  75%: 0.1054  -  Max: 15.2231
Skew: -11.5808  -  Kurtosis: 181.1436
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_217.png)

```
f_109
-----
Mean: 0.0112  -  Median: 0.2226  -  Std: 1.0339
Min: -13.8812  -  25%: -0.4599  -  50%: 0.2226  -  75%: 0.7076  -  Max: 6.6099
Skew: -1.4327  -  Kurtosis: 3.8281
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_219.png)

```
f_110
-----
Mean: -0.0054  -  Median: -0.0593  -  Std: 1.1118
Min: -33.3716  -  25%: -0.5396  -  50%: -0.0593  -  75%: 0.4690  -  Max: 16.7558
Skew: -0.8301  -  Kurtosis: 28.4744
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_221.png)

```
f_111
-----
Mean: -0.0027  -  Median: 0.1422  -  Std: 1.0895
Min: -10.6889  -  25%: -0.6202  -  50%: 0.1422  -  75%: 0.7549  -  Max: 6.8152
Skew: -0.7571  -  Kurtosis: 1.2517
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_223.png)

```
f_112
-----
Mean: 0.0028  -  Median: -0.1575  -  Std: 1.0479
Min: -5.7160  -  25%: -0.7573  -  50%: -0.1575  -  75%: 0.6024  -  Max: 7.8987
Skew: 0.7827  -  Kurtosis: 0.7654
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_225.png)

```
f_113
-----
Mean: -0.0209  -  Median: -0.2622  -  Std: 0.9768
Min: -7.5655  -  25%: -0.5706  -  50%: -0.2622  -  75%: 0.2553  -  Max: 45.9560
Skew: 4.4794  -  Kurtosis: 75.0031
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_227.png)

```
f_114
-----
Mean: 0.0020  -  Median: -0.0367  -  Std: 1.1070
Min: -12.3811  -  25%: -0.6937  -  50%: -0.0367  -  75%: 0.6564  -  Max: 9.3392
Skew: 0.2602  -  Kurtosis: 1.3117
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_229.png)

```
f_115
-----
Mean: -0.0038  -  Median: -0.1106  -  Std: 1.0525
Min: -62.3700  -  25%: -0.5032  -  50%: -0.1106  -  75%: 0.3286  -  Max: 94.2216
Skew: 15.2718  -  Kurtosis: 675.5988
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_231.png)

```
f_116
-----
Mean: -0.0109  -  Median: -0.1637  -  Std: 1.0131
Min: -6.9972  -  25%: -0.6917  -  50%: -0.1637  -  75%: 0.5240  -  Max: 21.4471
Skew: 1.2528  -  Kurtosis: 4.1805
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_233.png)

```
f_117
-----
Mean: -0.0083  -  Median: -0.1972  -  Std: 1.0016
Min: -31.6503  -  25%: -0.4727  -  50%: -0.1972  -  75%: 0.0581  -  Max: 87.5820
Skew: 11.3869  -  Kurtosis: 491.8940
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_235.png)

```
f_118
-----
Mean: -0.0013  -  Median: -0.0902  -  Std: 1.0714
Min: -45.6748  -  25%: -0.6181  -  50%: -0.0902  -  75%: 0.5709  -  Max: 11.7146
Skew: -0.2489  -  Kurtosis: 10.6331
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_237.png)

```
f_119
-----
Mean: 0.0136  -  Median: 0.3729  -  Std: 1.1000
Min: -9.4983  -  25%: -0.4132  -  50%: 0.3729  -  75%: 0.6494  -  Max: 5.9922
Skew: -1.3479  -  Kurtosis: 2.4576
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_239.png)

```
f_120
-----
Mean: -0.0040  -  Median: 0.3025  -  Std: 1.0178
Min: -11.0595  -  25%: -0.7174  -  50%: 0.3025  -  75%: 0.6946  -  Max: 4.5785
Skew: -0.8081  -  Kurtosis: 1.3728
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_241.png)

```
f_121
-----
Mean: -0.0149  -  Median: -0.2861  -  Std: 1.0000
Min: -9.0935  -  25%: -0.5597  -  50%: -0.2861  -  75%: 0.2548  -  Max: 18.6933
Skew: 2.4566  -  Kurtosis: 11.8564
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_243.png)

```
f_122
-----
Mean: -0.0098  -  Median: -0.0847  -  Std: 1.1068
Min: -15.0495  -  25%: -0.5517  -  50%: -0.0847  -  75%: 0.4673  -  Max: 48.7455
Skew: 2.3646  -  Kurtosis: 38.3465
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_245.png)

```
f_123
-----
Mean: -0.0109  -  Median: -0.1105  -  Std: 1.0061
Min: -19.6337  -  25%: -0.6407  -  50%: -0.1105  -  75%: 0.4986  -  Max: 45.7838
Skew: 5.0791  -  Kurtosis: 108.2449
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_247.png)

```
f_124
-----
Mean: -0.0001  -  Median: 0.0000  -  Std: 0.0484
Min: -10.4032  -  25%: -0.0000  -  50%: 0.0000  -  75%: 0.0000  -  Max: 34.0294
Skew: 548.6503  -  Kurtosis: 337461.6250
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_249.png)

```
f_125
-----
Mean: 0.0028  -  Median: 0.0209  -  Std: 1.0401
Min: -10.0809  -  25%: -0.6513  -  50%: 0.0209  -  75%: 0.6590  -  Max: 27.2795
Skew: 0.1096  -  Kurtosis: 1.9448
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_251.png)

```
f_126
-----
Mean: 0.0001  -  Median: -0.0460  -  Std: 1.0765
Min: -9.8482  -  25%: -0.5414  -  50%: -0.0460  -  75%: 0.6415  -  Max: 10.5036
Skew: -0.0997  -  Kurtosis: 1.2067
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_253.png)

```
f_127
-----
Mean: -0.0021  -  Median: 0.0048  -  Std: 1.0131
Min: -86.3378  -  25%: -0.0946  -  50%: 0.0048  -  75%: 0.0933  -  Max: 79.7563
Skew: -1.4597  -  Kurtosis: 941.1480
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_255.png)

```
f_128
-----
Mean: 0.0016  -  Median: 0.0898  -  Std: 1.1020
Min: -62.8390  -  25%: -0.4707  -  50%: 0.0898  -  75%: 0.5705  -  Max: 46.8688
Skew: -2.7491  -  Kurtosis: 48.1747
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_257.png)

```
f_129
-----
Mean: 0.0048  -  Median: 0.1166  -  Std: 1.0560
Min: -34.1543  -  25%: -0.5899  -  50%: 0.1166  -  75%: 0.7259  -  Max: 6.0911
Skew: -0.9161  -  Kurtosis: 4.9357
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_259.png)

```
f_130
-----
Mean: 0.0082  -  Median: 0.0792  -  Std: 1.1174
Min: -13.5667  -  25%: -0.5234  -  50%: 0.0792  -  75%: 0.6213  -  Max: 14.4541
Skew: -0.5431  -  Kurtosis: 4.2532
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_261.png)

```
f_131
-----
Mean: -0.0119  -  Median: -0.2152  -  Std: 0.9912
Min: -27.7068  -  25%: -0.5722  -  50%: -0.2152  -  75%: 0.2283  -  Max: 59.2735
Skew: 6.8161  -  Kurtosis: 205.9278
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_263.png)

```
f_132
-----
Mean: -0.0058  -  Median: -0.0429  -  Std: 1.1142
Min: -11.4331  -  25%: -0.7654  -  50%: -0.0429  -  75%: 0.7510  -  Max: 11.4711
Skew: 0.0411  -  Kurtosis: 0.7337
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_265.png)

```
f_133
-----
Mean: -0.0114  -  Median: -0.0266  -  Std: 1.0414
Min: -5.4257  -  25%: -0.8642  -  50%: -0.0266  -  75%: 0.8273  -  Max: 5.9043
Skew: 0.0502  -  Kurtosis: -0.8444
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_267.png)

```
f_134
-----
Mean: -0.0097  -  Median: -0.0229  -  Std: 1.0218
Min: -6.2701  -  25%: -0.8321  -  50%: -0.0229  -  75%: 0.8035  -  Max: 6.9731
Skew: 0.0379  -  Kurtosis: -0.8566
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_269.png)

```
f_135
-----
Mean: -0.0008  -  Median: -0.0114  -  Std: 1.0665
Min: -6.8656  -  25%: -0.7669  -  50%: -0.0114  -  75%: 0.7629  -  Max: 6.6044
Skew: 0.0177  -  Kurtosis: -0.3198
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_271.png)

```
f_136
-----
Mean: -0.0220  -  Median: -0.3036  -  Std: 0.9906
Min: -11.9715  -  25%: -0.5782  -  50%: -0.3036  -  75%: 0.2066  -  Max: 55.4699
Skew: 3.3020  -  Kurtosis: 22.0760
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_273.png)

```
f_137
-----
Mean: -0.0203  -  Median: -0.2464  -  Std: 0.9923
Min: -14.7507  -  25%: -0.4921  -  50%: -0.2464  -  75%: 0.1830  -  Max: 34.9401
Skew: 3.6621  -  Kurtosis: 34.5435
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_275.png)

```
f_138
-----
Mean: -0.0041  -  Median: -0.0728  -  Std: 1.0841
Min: -9.2837  -  25%: -0.7097  -  50%: -0.0728  -  75%: 0.6787  -  Max: 12.1628
Skew: 0.1276  -  Kurtosis: 1.4841
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_277.png)

```
f_139
-----
Mean: -0.0118  -  Median: -0.0960  -  Std: 1.0456
Min: -8.0609  -  25%: -0.7085  -  50%: -0.0960  -  75%: 0.6278  -  Max: 11.2357
Skew: 0.3545  -  Kurtosis: 1.2181
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_279.png)

```
f_140
-----
Mean: -0.0126  -  Median: -0.0162  -  Std: 1.0358
Min: -4.4354  -  25%: -0.8782  -  50%: -0.0162  -  75%: 0.8517  -  Max: 4.2737
Skew: 0.0088  -  Kurtosis: -0.9909
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_281.png)

```
f_141
-----
Mean: 0.0063  -  Median: -0.0526  -  Std: 1.0346
Min: -7.0399  -  25%: -0.6217  -  50%: -0.0526  -  75%: 0.6158  -  Max: 8.5520
Skew: 0.1043  -  Kurtosis: 1.4702
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_283.png)

```
f_142
-----
Mean: 0.0108  -  Median: 0.2495  -  Std: 1.0088
Min: -21.6628  -  25%: 0.1957  -  50%: 0.2495  -  75%: 0.2988  -  Max: 7.3912
Skew: -4.5959  -  Kurtosis: 26.8969
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_285.png)

```
f_143
-----
Mean: -0.0095  -  Median: -0.0667  -  Std: 1.0588
Min: -10.0204  -  25%: -0.7489  -  50%: -0.0667  -  75%: 0.7519  -  Max: 8.4157
Skew: 0.0399  -  Kurtosis: 0.4138
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_287.png)

```
f_144
-----
Mean: -0.0010  -  Median: -0.1585  -  Std: 1.0819
Min: -13.8389  -  25%: -0.6552  -  50%: -0.1585  -  75%: 0.5645  -  Max: 15.4763
Skew: 0.5078  -  Kurtosis: 3.5747
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_289.png)

```
f_145
-----
Mean: 0.0187  -  Median: 0.0922  -  Std: 1.0266
Min: -50.6579  -  25%: 0.0704  -  50%: 0.0922  -  75%: 0.1248  -  Max: 11.4596
Skew: -8.1678  -  Kurtosis: 100.6878
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_291.png)

```
f_146
-----
Mean: -0.0023  -  Median: 0.0037  -  Std: 1.0102
Min: -3.7990  -  25%: -0.8560  -  50%: 0.0037  -  75%: 0.8541  -  Max: 4.2372
Skew: -0.0132  -  Kurtosis: -1.0874
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_293.png)

```
f_147
-----
Mean: -0.0054  -  Median: -0.1954  -  Std: 1.0309
Min: -5.3405  -  25%: -0.8531  -  50%: -0.1954  -  75%: 0.8472  -  Max: 4.7427
Skew: 0.3387  -  Kurtosis: -0.7901
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_295.png)

```
f_148
-----
Mean: -0.0036  -  Median: -0.0023  -  Std: 1.0052
Min: -4.3926  -  25%: -0.6999  -  50%: -0.0023  -  75%: 0.6878  -  Max: 4.1398
Skew: 0.0035  -  Kurtosis: -0.5154
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_297.png)

```
f_149
-----
Mean: -0.0081  -  Median: -0.0995  -  Std: 1.1090
Min: -19.1982  -  25%: -0.6077  -  50%: -0.0995  -  75%: 0.4903  -  Max: 51.0214
Skew: 0.7292  -  Kurtosis: 6.2896
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_299.png)

```
f_150
-----
Mean: 0.0232  -  Median: 0.3324  -  Std: 0.9917
Min: -12.3214  -  25%: 0.0880  -  50%: 0.3324  -  75%: 0.4056  -  Max: 6.7936
Skew: -3.5383  -  Kurtosis: 15.3083
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_301.png)

```
f_151
-----
Mean: -0.0089  -  Median: -0.1460  -  Std: 0.8974
Min: -18.1290  -  25%: -0.3799  -  50%: -0.1460  -  75%: 0.0345  -  Max: 58.4704
Skew: 9.7553  -  Kurtosis: 271.6754
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_303.png)

```
f_152
-----
Mean: 0.0081  -  Median: 0.4657  -  Std: 1.0507
Min: -8.1173  -  25%: -0.4071  -  50%: 0.4657  -  75%: 0.6202  -  Max: 5.0976
Skew: -1.1869  -  Kurtosis: 0.3831
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_305.png)

```
f_153
-----
Mean: 0.0388  -  Median: 0.0730  -  Std: 0.8444
Min: -52.4662  -  25%: 0.0567  -  50%: 0.0730  -  75%: 0.0914  -  Max: 23.8206
Skew: -12.9356  -  Kurtosis: 266.5368
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_307.png)

```
f_154
-----
Mean: 0.0027  -  Median: -0.0793  -  Std: 1.1266
Min: -7.7861  -  25%: -0.7694  -  50%: -0.0793  -  75%: 0.7955  -  Max: 8.5341
Skew: 0.0661  -  Kurtosis: 0.4743
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_309.png)

```
f_155
-----
Mean: -0.0126  -  Median: -0.3112  -  Std: 1.0133
Min: -5.7723  -  25%: -0.6548  -  50%: -0.3112  -  75%: 0.2925  -  Max: 77.8047
Skew: 4.6065  -  Kurtosis: 160.0860
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_311.png)

```
f_156
-----
Mean: -0.0028  -  Median: 0.0569  -  Std: 1.1391
Min: -20.6954  -  25%: -0.7583  -  50%: 0.0569  -  75%: 0.7508  -  Max: 11.8746
Skew: -0.1549  -  Kurtosis: 1.9521
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_313.png)

```
f_157
-----
Mean: -0.0125  -  Median: 0.0015  -  Std: 1.0317
Min: -5.2213  -  25%: -0.6772  -  50%: 0.0015  -  75%: 0.6565  -  Max: 4.7843
Skew: -0.0360  -  Kurtosis: -0.3274
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_315.png)

```
f_158
-----
Mean: -0.0132  -  Median: -0.1378  -  Std: 1.0042
Min: -5.4247  -  25%: -0.7078  -  50%: -0.1378  -  75%: 0.5537  -  Max: 31.4943
Skew: 1.0654  -  Kurtosis: 8.9466
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_317.png)

```
f_159
-----
Mean: -0.0138  -  Median: -0.1621  -  Std: 0.9979
Min: -6.1227  -  25%: -0.6921  -  50%: -0.1621  -  75%: 0.5053  -  Max: 26.1429
Skew: 1.2315  -  Kurtosis: 4.6323
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_319.png)

```
f_160
-----
Mean: -0.0091  -  Median: -0.1949  -  Std: 1.0276
Min: -5.5080  -  25%: -0.7064  -  50%: -0.1949  -  75%: 0.4933  -  Max: 46.5437
Skew: 1.6068  -  Kurtosis: 22.0662
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_321.png)

```
f_161
-----
Mean: -0.0119  -  Median: -0.0160  -  Std: 1.0157
Min: -4.1658  -  25%: -0.8642  -  50%: -0.0160  -  75%: 0.8399  -  Max: 4.2333
Skew: 0.0086  -  Kurtosis: -1.0318
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_323.png)

```
f_162
-----
Mean: 0.0111  -  Median: -0.1062  -  Std: 1.0439
Min: -15.9403  -  25%: -0.6391  -  50%: -0.1062  -  75%: 0.5090  -  Max: 75.7261
Skew: 3.8328  -  Kurtosis: 120.8205
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_325.png)

```
f_163
-----
Mean: -0.0093  -  Median: -0.1691  -  Std: 1.0174
Min: -5.4055  -  25%: -0.8816  -  50%: -0.1691  -  75%: 0.9705  -  Max: 4.2386
Skew: 0.0021  -  Kurtosis: -1.1293
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_327.png)

```
f_164
-----
Mean: 0.0106  -  Median: 0.0242  -  Std: 1.0042
Min: -3.2804  -  25%: -0.8455  -  50%: 0.0242  -  75%: 0.8687  -  Max: 3.1756
Skew: -0.0279  -  Kurtosis: -1.1193
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_329.png)

```
f_165
-----
Mean: -0.0156  -  Median: -0.1084  -  Std: 1.0346
Min: -16.1574  -  25%: -0.2749  -  50%: -0.1084  -  75%: 0.1191  -  Max: 64.3399
Skew: 2.7496  -  Kurtosis: 37.4143
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_331.png)

```
f_166
-----
Mean: 0.0147  -  Median: 0.0958  -  Std: 1.0415
Min: -8.4342  -  25%: -0.5966  -  50%: 0.0958  -  75%: 0.6948  -  Max: 5.7025
Skew: -0.5508  -  Kurtosis: 1.2314
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_333.png)

```
f_167
-----
Mean: -0.0111  -  Median: -0.2890  -  Std: 1.0340
Min: -6.7978  -  25%: -0.6537  -  50%: -0.2890  -  75%: 0.3202  -  Max: 25.4334
Skew: 2.7418  -  Kurtosis: 16.0408
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_335.png)

```
f_168
-----
Mean: -0.0051  -  Median: -0.3938  -  Std: 1.0227
Min: -14.1596  -  25%: -0.6135  -  50%: -0.3938  -  75%: 0.8621  -  Max: 9.0329
Skew: -0.0888  -  Kurtosis: 1.9287
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_337.png)

```
f_169
-----
Mean: 0.0102  -  Median: -0.2936  -  Std: 1.0390
Min: -7.9361  -  25%: -0.6944  -  50%: -0.2936  -  75%: 0.4114  -  Max: 23.7696
Skew: 2.1564  -  Kurtosis: 8.9535
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_339.png)

```
f_170
-----
Mean: -0.1256  -  Median: 0.0000  -  Std: 0.6391
Min: -4.2295  -  25%: -0.4585  -  50%: 0.0000  -  75%: 0.0588  -  Max: 4.2051
Skew: -0.1232  -  Kurtosis: 0.4916
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_341.png)

```
f_171
-----
Mean: -0.0082  -  Median: -0.0886  -  Std: 1.0747
Min: -8.2756  -  25%: -0.7191  -  50%: -0.0886  -  75%: 0.6641  -  Max: 14.0777
Skew: 0.3317  -  Kurtosis: 1.1644
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_343.png)

```
f_172
-----
Mean: -0.0221  -  Median: -0.2640  -  Std: 0.9771
Min: -8.2624  -  25%: -0.5029  -  50%: -0.2640  -  75%: 0.1104  -  Max: 79.7142
Skew: 6.9330  -  Kurtosis: 211.9505
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_345.png)

```
f_173
-----
Mean: -0.0090  -  Median: -0.2529  -  Std: 1.0151
Min: -5.5647  -  25%: -0.6877  -  50%: -0.2529  -  75%: 0.3953  -  Max: 28.5141
Skew: 2.2003  -  Kurtosis: 11.7382
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_347.png)

```
f_174
-----
Mean: 0.0006  -  Median: -0.1864  -  Std: 1.1276
Min: -7.8700  -  25%: -0.7094  -  50%: -0.1864  -  75%: 0.7416  -  Max: 47.1063
Skew: 0.6383  -  Kurtosis: 4.3604
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_349.png)

```
f_175
-----
Mean: -0.4781  -  Median: -0.5983  -  Std: 0.6749
Min: -56.5509  -  25%: -1.0000  -  50%: -0.5983  -  75%: 0.0000  -  Max: 27.9929
Skew: -3.6631  -  Kurtosis: 303.0509
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_351.png)

```
f_176
-----
Mean: -0.0043  -  Median: 0.2807  -  Std: 1.1423
Min: -9.6675  -  25%: -0.8956  -  50%: 0.2807  -  75%: 0.8338  -  Max: 6.5403
Skew: -0.3918  -  Kurtosis: -0.0054
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_353.png)

```
f_177
-----
Mean: 0.0051  -  Median: 0.0498  -  Std: 1.0940
Min: -9.2915  -  25%: -0.7191  -  50%: 0.0498  -  75%: 0.7379  -  Max: 7.4673
Skew: -0.0799  -  Kurtosis: 0.5098
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_355.png)

```
f_178
-----
Mean: 0.0130  -  Median: 0.0437  -  Std: 1.0639
Min: -11.6910  -  25%: -0.5945  -  50%: 0.0437  -  75%: 0.6667  -  Max: 8.1853
Skew: -0.4367  -  Kurtosis: 2.1179
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_357.png)

```
f_179
-----
Mean: -0.0071  -  Median: -0.0928  -  Std: 1.0686
Min: -45.6490  -  25%: -0.3970  -  50%: -0.0928  -  75%: 0.2904  -  Max: 39.7430
Skew: 1.2151  -  Kurtosis: 30.5551
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_359.png)

```
f_180
-----
Mean: 0.0090  -  Median: 0.0128  -  Std: 1.0061
Min: -5.1178  -  25%: -0.6910  -  50%: 0.0128  -  75%: 0.7116  -  Max: 5.2345
Skew: -0.0111  -  Kurtosis: -0.1934
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_361.png)

```
f_181
-----
Mean: 0.0112  -  Median: -0.1683  -  Std: 1.0779
Min: -6.0520  -  25%: -0.8287  -  50%: -0.1683  -  75%: 0.7094  -  Max: 7.6521
Skew: 0.6202  -  Kurtosis: 0.0511
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_363.png)

```
f_182
-----
Mean: 0.3917  -  Median: 0.4995  -  Std: 0.7506
Min: -4.3532  -  25%: 0.0000  -  50%: 0.4995  -  75%: 1.0000  -  Max: 4.5236
Skew: -0.5352  -  Kurtosis: -0.2185
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_365.png)

```
f_183
-----
Mean: 0.0031  -  Median: 0.0018  -  Std: 0.9490
Min: -65.1881  -  25%: -0.0216  -  50%: 0.0018  -  75%: 0.0282  -  Max: 65.5111
Skew: 1.9182  -  Kurtosis: 831.3126
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_367.png)

```
f_184
-----
Mean: -0.0060  -  Median: 0.0289  -  Std: 1.0306
Min: -35.3542  -  25%: -0.6467  -  50%: 0.0289  -  75%: 0.7107  -  Max: 12.2592
Skew: -1.3414  -  Kurtosis: 17.1370
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_369.png)

```
f_185
-----
Mean: -0.0169  -  Median: -0.0088  -  Std: 1.0636
Min: -6.3683  -  25%: -0.7454  -  50%: -0.0088  -  75%: 0.6248  -  Max: 8.8749
Skew: 0.2579  -  Kurtosis: 0.4709
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_371.png)

```
f_186
-----
Mean: -0.0102  -  Median: -0.2694  -  Std: 0.9674
Min: -23.2229  -  25%: -0.6346  -  50%: -0.2694  -  75%: 0.3178  -  Max: 23.0673
Skew: 2.1208  -  Kurtosis: 7.9971
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_373.png)

```
f_187
-----
Mean: -0.0058  -  Median: -0.0730  -  Std: 1.0930
Min: -6.9549  -  25%: -0.9418  -  50%: -0.0730  -  75%: 0.9319  -  Max: 6.6286
Skew: 0.1112  -  Kurtosis: -0.8169
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_375.png)

```
f_188
-----
Mean: -0.0194  -  Median: -0.2569  -  Std: 1.0009
Min: -6.8619  -  25%: -0.5800  -  50%: -0.2569  -  75%: 0.2462  -  Max: 51.7645
Skew: 3.7865  -  Kurtosis: 44.3326
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_377.png)

```
f_189
-----
Mean: -0.0065  -  Median: -0.1899  -  Std: 1.0462
Min: -6.2250  -  25%: -0.6936  -  50%: -0.1899  -  75%: 0.4749  -  Max: 19.5765
Skew: 1.4545  -  Kurtosis: 5.2316
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_379.png)

```
f_190
-----
Mean: -0.0009  -  Median: 0.1011  -  Std: 1.0556
Min: -6.2053  -  25%: -0.7479  -  50%: 0.1011  -  75%: 0.6993  -  Max: 12.6941
Skew: 0.0121  -  Kurtosis: 0.1166
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_381.png)

```
f_191
-----
Mean: -0.0101  -  Median: 0.0000  -  Std: 0.9135
Min: -7.0369  -  25%: -0.5351  -  50%: 0.0000  -  75%: 0.4395  -  Max: 17.8740
Skew: 0.4393  -  Kurtosis: 2.9354
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_383.png)

```
f_192
-----
Mean: 0.0165  -  Median: 0.1326  -  Std: 1.0063
Min: -9.5603  -  25%: -0.4015  -  50%: 0.1326  -  75%: 0.5701  -  Max: 12.3438
Skew: -1.1479  -  Kurtosis: 5.0369
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_385.png)

```
f_193
-----
Mean: -0.0229  -  Median: -0.2484  -  Std: 0.9726
Min: -7.3174  -  25%: -0.5570  -  50%: -0.2484  -  75%: 0.2223  -  Max: 75.4884
Skew: 5.4553  -  Kurtosis: 154.2198
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_387.png)

```
f_194
-----
Mean: 0.0093  -  Median: 0.0399  -  Std: 1.0808
Min: -5.6196  -  25%: -0.7517  -  50%: 0.0399  -  75%: 0.7854  -  Max: 7.2686
Skew: -0.0909  -  Kurtosis: -0.3071
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_389.png)

```
f_195
-----
Mean: -0.0022  -  Median: -0.0068  -  Std: 1.0991
Min: -9.2084  -  25%: -0.7373  -  50%: -0.0068  -  75%: 0.7317  -  Max: 9.2393
Skew: -0.0224  -  Kurtosis: 0.6019
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_391.png)

```
f_196
-----
Mean: 0.0065  -  Median: -0.0176  -  Std: 1.1071
Min: -36.3240  -  25%: -0.4913  -  50%: -0.0176  -  75%: 0.4760  -  Max: 52.0245
Skew: 0.2665  -  Kurtosis: 9.6687
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_393.png)

```
f_197
-----
Mean: -0.0200  -  Median: -0.2831  -  Std: 1.0762
Min: -9.3545  -  25%: -0.5393  -  50%: -0.2831  -  75%: 0.2493  -  Max: 59.9578
Skew: 2.5587  -  Kurtosis: 15.7319
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_395.png)

```
f_198
-----
Mean: 0.0113  -  Median: 0.1849  -  Std: 1.0775
Min: -9.0244  -  25%: -0.6669  -  50%: 0.1849  -  75%: 0.8104  -  Max: 6.8733
Skew: -0.5599  -  Kurtosis: 0.0244
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_397.png)

```
f_199
-----
Mean: -0.0010  -  Median: 0.0000  -  Std: 1.0420
Min: -8.8161  -  25%: -0.6657  -  50%: 0.0000  -  75%: 0.6423  -  Max: 8.6496
Skew: 0.1202  -  Kurtosis: 0.7725
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_399.png)

```
f_200
-----
Mean: -0.0369  -  Median: 0.0000  -  Std: 0.7914
Min: -11.7403  -  25%: -0.4323  -  50%: 0.0000  -  75%: 0.2945  -  Max: 86.6484
Skew: 12.0562  -  Kurtosis: 875.2459
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_401.png)

```
f_201
-----
Mean: -0.0074  -  Median: -0.0266  -  Std: 1.0595
Min: -35.4576  -  25%: -0.4415  -  50%: -0.0266  -  75%: 0.4225  -  Max: 42.1553
Skew: 1.8447  -  Kurtosis: 62.0081
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_403.png)

```
f_202
-----
Mean: -0.0080  -  Median: -0.0582  -  Std: 1.0564
Min: -28.0177  -  25%: -0.3948  -  50%: -0.0582  -  75%: 0.3312  -  Max: 47.9185
Skew: 0.6106  -  Kurtosis: 45.3021
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_405.png)

```
f_203
-----
Mean: -0.0133  -  Median: -0.2266  -  Std: 1.0009
Min: -5.4844  -  25%: -0.6889  -  50%: -0.2266  -  75%: 0.4197  -  Max: 22.3154
Skew: 1.9959  -  Kurtosis: 9.1932
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_407.png)

```
f_204
-----
Mean: 0.0013  -  Median: 0.1668  -  Std: 1.0430
Min: -43.3910  -  25%: -0.1067  -  50%: 0.1668  -  75%: 0.3712  -  Max: 9.7633
Skew: -5.4043  -  Kurtosis: 78.8448
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_409.png)

```
f_205
-----
Mean: 0.0042  -  Median: 0.0116  -  Std: 1.0073
Min: -6.8698  -  25%: -0.6761  -  50%: 0.0116  -  75%: 0.6751  -  Max: 19.8350
Skew: 0.0479  -  Kurtosis: 0.5561
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_411.png)

```
f_206
-----
Mean: 0.0001  -  Median: 0.0099  -  Std: 1.0730
Min: -11.0336  -  25%: -0.6505  -  50%: 0.0099  -  75%: 0.6441  -  Max: 10.8770
Skew: 0.0861  -  Kurtosis: 1.6685
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_413.png)

```
f_207
-----
Mean: -0.0105  -  Median: -0.0457  -  Std: 1.0072
Min: -6.5908  -  25%: -0.5035  -  50%: -0.0457  -  75%: 0.4556  -  Max: 17.9975
Skew: 0.9969  -  Kurtosis: 5.6204
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_415.png)

```
f_208
-----
Mean: -0.0030  -  Median: 0.1081  -  Std: 1.0494
Min: -23.6346  -  25%: -0.5177  -  50%: 0.1081  -  75%: 0.6438  -  Max: 14.9698
Skew: -0.8952  -  Kurtosis: 5.1639
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_417.png)

```
f_209
-----
Mean: -0.0096  -  Median: -0.0216  -  Std: 1.0850
Min: -41.6647  -  25%: -0.4946  -  50%: -0.0216  -  75%: 0.4638  -  Max: 43.0806
Skew: 0.2005  -  Kurtosis: 51.8425
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_419.png)

```
f_210
-----
Mean: -0.0056  -  Median: -0.2970  -  Std: 1.0061
Min: -16.2018  -  25%: -0.4013  -  50%: -0.2970  -  75%: 0.0747  -  Max: 23.5734
Skew: 4.1535  -  Kurtosis: 39.3228
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_421.png)

```
f_211
-----
Mean: -0.0024  -  Median: 0.1350  -  Std: 1.0187
Min: -4.5330  -  25%: -0.7775  -  50%: 0.1350  -  75%: 0.8641  -  Max: 4.0372
Skew: -0.3918  -  Kurtosis: -0.8167
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_423.png)

```
f_212
-----
Mean: -0.0028  -  Median: -0.1319  -  Std: 1.0117
Min: -5.4074  -  25%: -0.7307  -  50%: -0.1319  -  75%: 0.5898  -  Max: 8.7332
Skew: 0.7280  -  Kurtosis: 0.8290
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_425.png)

```
f_213
-----
Mean: -0.0006  -  Median: -0.0111  -  Std: 1.1398
Min: -9.8643  -  25%: -0.7583  -  50%: -0.0111  -  75%: 0.7625  -  Max: 8.6885
Skew: -0.0161  -  Kurtosis: 0.5644
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_427.png)

```
f_214
-----
Mean: -0.0224  -  Median: -0.2041  -  Std: 1.0092
Min: -6.8143  -  25%: -0.7057  -  50%: -0.2041  -  75%: 0.4692  -  Max: 62.4495
Skew: 1.2521  -  Kurtosis: 7.2422
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_429.png)

```
f_215
-----
Mean: -0.0065  -  Median: -0.0495  -  Std: 1.0104
Min: -11.7265  -  25%: -0.4855  -  50%: -0.0495  -  75%: 0.4040  -  Max: 52.0995
Skew: 7.2524  -  Kurtosis: 131.1801
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_431.png)

```
f_216
-----
Mean: -0.0059  -  Median: -0.0527  -  Std: 1.0337
Min: -6.3557  -  25%: -0.7044  -  50%: -0.0527  -  75%: 0.6481  -  Max: 7.1191
Skew: 0.2539  -  Kurtosis: 0.5657
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_433.png)

```
f_217
-----
Mean: -0.0069  -  Median: -0.0097  -  Std: 1.0400
Min: -5.5343  -  25%: -0.8658  -  50%: -0.0097  -  75%: 0.8492  -  Max: 5.2208
Skew: 0.0112  -  Kurtosis: -0.9109
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_435.png)

```
f_218
-----
Mean: -0.0129  -  Median: -0.3425  -  Std: 1.0046
Min: -4.4790  -  25%: -0.6637  -  50%: -0.3425  -  75%: 0.3182  -  Max: 16.4958
Skew: 2.2683  -  Kurtosis: 8.3815
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_437.png)

```
f_219
-----
Mean: 0.0015  -  Median: 0.0917  -  Std: 1.1358
Min: -15.8113  -  25%: -0.7831  -  50%: 0.0917  -  75%: 0.7744  -  Max: 49.7796
Skew: -0.1101  -  Kurtosis: 2.5243
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_439.png)

```
f_220
-----
Mean: -0.0034  -  Median: -0.1788  -  Std: 1.0434
Min: -14.9599  -  25%: -0.6650  -  50%: -0.1788  -  75%: 0.4558  -  Max: 20.1575
Skew: 1.7211  -  Kurtosis: 8.7465
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_441.png)

```
f_221
-----
Mean: 0.0001  -  Median: -0.0348  -  Std: 1.0831
Min: -6.3488  -  25%: -0.7781  -  50%: -0.0348  -  75%: 0.7892  -  Max: 6.9409
Skew: 0.0207  -  Kurtosis: -0.2600
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_443.png)

```
f_222
-----
Mean: 0.0087  -  Median: -0.0166  -  Std: 1.1245
Min: -7.5654  -  25%: -0.7903  -  50%: -0.0166  -  75%: 0.7987  -  Max: 7.8302
Skew: 0.0591  -  Kurtosis: 0.1690
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_445.png)

```
f_223
-----
Mean: -0.0051  -  Median: 0.0001  -  Std: 1.0823
Min: -12.7688  -  25%: -0.6380  -  50%: 0.0001  -  75%: 0.6186  -  Max: 10.5167
Skew: -0.0661  -  Kurtosis: 2.0354
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_447.png)

```
f_224
-----
Mean: -0.0161  -  Median: -0.1227  -  Std: 0.9987
Min: -18.7040  -  25%: -0.2271  -  50%: -0.1227  -  75%: 0.0574  -  Max: 59.5552
Skew: 16.2533  -  Kurtosis: 453.3520
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_449.png)

```
f_225
-----
Mean: 0.0262  -  Median: 0.1843  -  Std: 1.0538
Min: -18.7255  -  25%: 0.0987  -  50%: 0.1843  -  75%: 0.2995  -  Max: 12.4644
Skew: -4.0312  -  Kurtosis: 24.4019
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_451.png)

```
f_226
-----
Mean: -0.0132  -  Median: -0.0561  -  Std: 1.0270
Min: -5.3073  -  25%: -0.8289  -  50%: -0.0561  -  75%: 0.7327  -  Max: 7.3734
Skew: 0.2981  -  Kurtosis: -0.2898
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_453.png)

```
f_227
-----
Mean: -0.0007  -  Median: 0.0914  -  Std: 1.1280
Min: -7.3589  -  25%: -0.8292  -  50%: 0.0914  -  75%: 0.8734  -  Max: 6.2552
Skew: -0.2312  -  Kurtosis: -0.2901
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_455.png)

```
f_228
-----
Mean: -0.0021  -  Median: -0.2634  -  Std: 1.0046
Min: -13.7488  -  25%: -0.5307  -  50%: -0.2634  -  75%: 0.2135  -  Max: 58.8822
Skew: 8.1438  -  Kurtosis: 207.7540
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_457.png)

```
f_229
-----
Mean: -0.0045  -  Median: -0.6345  -  Std: 1.0286
Min: -4.7562  -  25%: -0.9469  -  50%: -0.6345  -  75%: 1.0284  -  Max: 4.9414
Skew: 0.2611  -  Kurtosis: -1.5859
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_459.png)

```
f_230
-----
Mean: -0.0118  -  Median: -0.0175  -  Std: 1.0281
Min: -5.2339  -  25%: -0.8694  -  50%: -0.0175  -  75%: 0.8415  -  Max: 4.9455
Skew: 0.0188  -  Kurtosis: -0.9743
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_461.png)

```
f_231
-----
Mean: -0.0067  -  Median: -0.2550  -  Std: 1.0740
Min: -7.5884  -  25%: -0.2680  -  50%: -0.2550  -  75%: -0.1445  -  Max: 16.7082
Skew: 4.0955  -  Kurtosis: 22.2722
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_463.png)

```
f_232
-----
Mean: -0.0088  -  Median: -0.2564  -  Std: 1.1036
Min: -10.6010  -  25%: -0.6409  -  50%: -0.2564  -  75%: 0.3623  -  Max: 19.0311
Skew: 2.0800  -  Kurtosis: 9.0585
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_465.png)

```
f_233
-----
Mean: -0.0114  -  Median: -0.0675  -  Std: 1.0471
Min: -28.0698  -  25%: -0.5143  -  50%: -0.0675  -  75%: 0.4374  -  Max: 45.2901
Skew: -0.4760  -  Kurtosis: 20.6330
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_467.png)

```
f_234
-----
Mean: -0.0025  -  Median: -0.0750  -  Std: 1.0263
Min: -4.5066  -  25%: -0.8012  -  50%: -0.0750  -  75%: 0.8065  -  Max: 6.9935
Skew: 0.1913  -  Kurtosis: -0.4616
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_469.png)

```
f_235
-----
Mean: -0.0097  -  Median: -0.1306  -  Std: 1.0979
Min: -28.2427  -  25%: -0.5848  -  50%: -0.1306  -  75%: 0.4322  -  Max: 31.3971
Skew: 0.5551  -  Kurtosis: 23.5875
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_471.png)

```
f_236
-----
Mean: -0.0082  -  Median: -0.1612  -  Std: 1.0127
Min: -31.9172  -  25%: -0.3226  -  50%: -0.1612  -  75%: 0.0439  -  Max: 59.0155
Skew: 13.0810  -  Kurtosis: 330.7409
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_473.png)

```
f_237
-----
Mean: -0.0064  -  Median: -0.0902  -  Std: 1.0696
Min: -7.8730  -  25%: -0.6564  -  50%: -0.0902  -  75%: 0.5154  -  Max: 17.8116
Skew: 1.0526  -  Kurtosis: 4.2839
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_475.png)

```
f_238
-----
Mean: 0.0022  -  Median: 0.3367  -  Std: 1.0350
Min: -6.6947  -  25%: -0.6682  -  50%: 0.3367  -  75%: 0.7257  -  Max: 4.9175
Skew: -0.8711  -  Kurtosis: 0.1961
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_477.png)

```
f_239
-----
Mean: -0.0095  -  Median: -0.1682  -  Std: 1.0754
Min: -7.0148  -  25%: -0.7221  -  50%: -0.1682  -  75%: 0.5866  -  Max: 9.1036
Skew: 0.7266  -  Kurtosis: 1.3152
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_479.png)

```
f_240
-----
Mean: -0.0193  -  Median: -0.2861  -  Std: 0.9783
Min: -7.8472  -  25%: -0.5566  -  50%: -0.2861  -  75%: 0.2175  -  Max: 24.2303
Skew: 4.0463  -  Kurtosis: 32.4151
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_481.png)

```
f_241
-----
Mean: 0.0203  -  Median: 0.1600  -  Std: 1.0241
Min: -19.6386  -  25%: 0.1174  -  50%: 0.1600  -  75%: 0.2203  -  Max: 8.9669
Skew: -5.2877  -  Kurtosis: 37.6520
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_483.png)

```
f_242
-----
Mean: 0.0128  -  Median: 0.0197  -  Std: 1.0211
Min: -6.2325  -  25%: -0.8174  -  50%: 0.0197  -  75%: 0.8506  -  Max: 4.7929
Skew: -0.0654  -  Kurtosis: -0.6935
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_485.png)

```
f_243
-----
Mean: -0.0149  -  Median: -0.2846  -  Std: 1.0795
Min: -7.8144  -  25%: -0.6783  -  50%: -0.2846  -  75%: 0.5491  -  Max: 13.8959
Skew: 1.0969  -  Kurtosis: 2.8546
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_487.png)

```
f_244
-----
Mean: 0.0038  -  Median: 0.1860  -  Std: 1.0972
Min: -27.4228  -  25%: -0.5191  -  50%: 0.1860  -  75%: 0.7337  -  Max: 10.3458
Skew: -1.5979  -  Kurtosis: 8.7964
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_489.png)

```
f_245
-----
Mean: -0.0162  -  Median: -0.3010  -  Std: 0.9843
Min: -4.5125  -  25%: -0.5290  -  50%: -0.3010  -  75%: 0.1519  -  Max: 56.8245
Skew: 6.1047  -  Kurtosis: 114.1208
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_491.png)

```
f_246
-----
Mean: 0.5633  -  Median: 0.0000  -  Std: 0.8509
Min: -4.3303  -  25%: 0.0000  -  50%: 0.0000  -  75%: 1.3661  -  Max: 6.9796
Skew: 1.0269  -  Kurtosis: 0.3290
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_493.png)

```
f_247
-----
Mean: -0.0088  -  Median: 0.0038  -  Std: 1.0129
Min: -5.0526  -  25%: -0.8121  -  50%: 0.0038  -  75%: 0.7326  -  Max: 4.7114
Skew: 0.0889  -  Kurtosis: -0.7247
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_495.png)

```
f_248
-----
Mean: 0.0137  -  Median: 0.0092  -  Std: 1.0382
Min: -5.5084  -  25%: -0.8498  -  50%: 0.0092  -  75%: 0.8764  -  Max: 4.5597
Skew: 0.0140  -  Kurtosis: -0.9461
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_497.png)

```
f_249
-----
Mean: 0.0015  -  Median: -0.0543  -  Std: 1.0229
Min: -21.1617  -  25%: -0.1751  -  50%: -0.0543  -  75%: 0.1040  -  Max: 55.6336
Skew: 2.2465  -  Kurtosis: 43.4285
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_499.png)

```
f_250
-----
Mean: -0.0269  -  Median: -0.2657  -  Std: 0.9995
Min: -7.5873  -  25%: -0.6003  -  50%: -0.2657  -  75%: 0.2577  -  Max: 38.3146
Skew: 3.2852  -  Kurtosis: 26.7433
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_501.png)

```
f_251
-----
Mean: -0.0100  -  Median: -0.2394  -  Std: 1.0466
Min: -15.9806  -  25%: -0.3624  -  50%: -0.2394  -  75%: 0.0027  -  Max: 64.2756
Skew: 7.3823  -  Kurtosis: 110.9861
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_503.png)

```
f_252
-----
Mean: -0.0145  -  Median: -0.0124  -  Std: 1.0437
Min: -5.3853  -  25%: -0.8708  -  50%: -0.0124  -  75%: 0.8441  -  Max: 5.1076
Skew: -0.0108  -  Kurtosis: -0.8992
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_505.png)

```
f_253
-----
Mean: -0.0095  -  Median: -0.1243  -  Std: 1.0543
Min: -7.7795  -  25%: -0.7237  -  50%: -0.1243  -  75%: 0.5949  -  Max: 11.1929
Skew: 0.7099  -  Kurtosis: 1.7101
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_507.png)

```
f_254
-----
Mean: -0.0040  -  Median: -0.0017  -  Std: 1.0219
Min: -4.1904  -  25%: -0.8662  -  50%: -0.0017  -  75%: 0.8578  -  Max: 4.4823
Skew: -0.0032  -  Kurtosis: -1.0576
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_509.png)

```
f_255
-----
Mean: -0.0164  -  Median: -0.3149  -  Std: 0.9961
Min: -7.8075  -  25%: -0.6168  -  50%: -0.3149  -  75%: 0.2576  -  Max: 19.8551
Skew: 3.0248  -  Kurtosis: 16.6519
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_511.png)

```
f_256
-----
Mean: -0.0056  -  Median: -0.1508  -  Std: 1.0943
Min: -7.6716  -  25%: -0.7128  -  50%: -0.1508  -  75%: 0.5915  -  Max: 12.2479
Skew: 0.7282  -  Kurtosis: 1.6858
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_513.png)

```
f_257
-----
Mean: -0.0037  -  Median: 0.1239  -  Std: 1.1066
Min: -18.0110  -  25%: -0.7737  -  50%: 0.1239  -  75%: 0.7065  -  Max: 8.4876
Skew: -0.3813  -  Kurtosis: 2.7455
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_515.png)

```
f_258
-----
Mean: -0.0018  -  Median: 0.0000  -  Std: 1.0407
Min: -6.0461  -  25%: -0.7153  -  50%: 0.0000  -  75%: 0.6905  -  Max: 7.2039
Skew: 0.0892  -  Kurtosis: 0.0740
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_517.png)

```
f_259
-----
Mean: -0.0048  -  Median: -0.0225  -  Std: 1.1118
Min: -7.0880  -  25%: -0.7756  -  50%: -0.0225  -  75%: 0.7657  -  Max: 7.2434
Skew: 0.0314  -  Kurtosis: -0.0300
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_519.png)

```
f_260
-----
Mean: -0.0055  -  Median: -0.1566  -  Std: 1.0840
Min: -7.1348  -  25%: -0.7313  -  50%: -0.1566  -  75%: 0.5929  -  Max: 20.2740
Skew: 0.8805  -  Kurtosis: 2.4179
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_521.png)

```
f_261
-----
Mean: 0.0004  -  Median: 0.1286  -  Std: 1.0315
Min: -6.4765  -  25%: -0.6266  -  50%: 0.1286  -  75%: 0.7986  -  Max: 4.0442
Skew: -0.6776  -  Kurtosis: 0.2782
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_523.png)

```
f_262
-----
Mean: -0.0141  -  Median: -0.2840  -  Std: 0.9879
Min: -3.2798  -  25%: -0.5437  -  50%: -0.2840  -  75%: 0.1980  -  Max: 54.1854
Skew: 7.7994  -  Kurtosis: 192.5882
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_525.png)

```
f_263
-----
Mean: 0.0103  -  Median: 0.0116  -  Std: 1.0694
Min: -6.0141  -  25%: -0.9943  -  50%: 0.0116  -  75%: 1.0044  -  Max: 5.7838
Skew: -0.0097  -  Kurtosis: -1.4305
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_527.png)

```
f_264
-----
Mean: 0.0204  -  Median: 0.2508  -  Std: 1.0153
Min: -24.3764  -  25%: -0.3847  -  50%: 0.2508  -  75%: 0.6555  -  Max: 6.9202
Skew: -2.0065  -  Kurtosis: 9.4746
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_529.png)

```
f_265
-----
Mean: -0.0189  -  Median: -0.2655  -  Std: 1.0215
Min: -10.0710  -  25%: -0.5634  -  50%: -0.2655  -  75%: 0.2168  -  Max: 36.2591
Skew: 4.0662  -  Kurtosis: 36.3888
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_531.png)

```
f_266
-----
Mean: 0.0121  -  Median: 0.0446  -  Std: 1.0447
Min: -13.9400  -  25%: -0.5940  -  50%: 0.0446  -  75%: 0.6643  -  Max: 8.0235
Skew: -0.6788  -  Kurtosis: 4.3821
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_533.png)

```
f_267
-----
Mean: 0.0098  -  Median: 0.1006  -  Std: 1.0267
Min: -7.5577  -  25%: -0.7054  -  50%: 0.1006  -  75%: 0.7311  -  Max: 12.2774
Skew: -0.1519  -  Kurtosis: 0.9661
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_535.png)

```
f_268
-----
Mean: -0.0165  -  Median: -0.1217  -  Std: 1.0359
Min: -5.0062  -  25%: -0.6884  -  50%: -0.1217  -  75%: 0.6015  -  Max: 6.2510
Skew: 0.4342  -  Kurtosis: 0.2887
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_537.png)

```
f_269
-----
Mean: -0.0000  -  Median: -0.1967  -  Std: 1.0471
Min: -15.6181  -  25%: -0.5517  -  50%: -0.1967  -  75%: 0.3868  -  Max: 70.9014
Skew: 3.9136  -  Kurtosis: 106.7898
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_539.png)

```
f_270
-----
Mean: 0.0209  -  Median: 0.2763  -  Std: 1.0064
Min: -21.1849  -  25%: -0.3481  -  50%: 0.2763  -  75%: 0.6466  -  Max: 9.6671
Skew: -2.3490  -  Kurtosis: 12.5609
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_541.png)

```
f_271
-----
Mean: -0.0160  -  Median: -0.2254  -  Std: 0.9819
Min: -13.2336  -  25%: -0.3648  -  50%: -0.2254  -  75%: 0.0078  -  Max: 56.3154
Skew: 10.0865  -  Kurtosis: 202.5123
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_543.png)

```
f_272
-----
Mean: -0.0294  -  Median: 0.0000  -  Std: 0.7614
Min: -3.9356  -  25%: -0.5283  -  50%: 0.0000  -  75%: 0.4473  -  Max: 5.4973
Skew: -0.1238  -  Kurtosis: -0.3123
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_545.png)

```
f_273
-----
Mean: -0.0111  -  Median: -0.0312  -  Std: 1.0400
Min: -6.7974  -  25%: -0.7414  -  50%: -0.0312  -  75%: 0.7007  -  Max: 6.2108
Skew: 0.0925  -  Kurtosis: -0.1620
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_547.png)

```
f_274
-----
Mean: -0.0208  -  Median: -0.2871  -  Std: 0.9758
Min: -4.8553  -  25%: -0.5444  -  50%: -0.2871  -  75%: 0.2034  -  Max: 22.1939
Skew: 3.7276  -  Kurtosis: 25.8005
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_549.png)

```
f_275
-----
Mean: 0.0016  -  Median: -0.2461  -  Std: 1.0210
Min: -4.8290  -  25%: -0.5083  -  50%: -0.2461  -  75%: 0.2119  -  Max: 25.8859
Skew: 4.2810  -  Kurtosis: 29.2717
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_551.png)

```
f_276
-----
Mean: 0.0172  -  Median: 0.1074  -  Std: 0.9943
Min: -8.5633  -  25%: -0.4941  -  50%: 0.1074  -  75%: 0.6600  -  Max: 4.8741
Skew: -0.8894  -  Kurtosis: 2.1704
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_553.png)

```
f_277
-----
Mean: -0.0113  -  Median: -0.0635  -  Std: 1.0355
Min: -6.4898  -  25%: -0.7899  -  50%: -0.0635  -  75%: 0.6899  -  Max: 60.6227
Skew: 0.4270  -  Kurtosis: 3.9823
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_555.png)

```
f_278
-----
Mean: -0.0031  -  Median: 0.2120  -  Std: 1.0572
Min: -27.8035  -  25%: -0.5965  -  50%: 0.2120  -  75%: 0.6470  -  Max: 7.2471
Skew: -1.2570  -  Kurtosis: 6.6646
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_557.png)

```
f_279
-----
Mean: 0.0009  -  Median: 0.0205  -  Std: 1.1324
Min: -14.1980  -  25%: -0.7204  -  50%: 0.0205  -  75%: 0.6602  -  Max: 20.7472
Skew: 0.2831  -  Kurtosis: 3.2363
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_559.png)

```
f_280
-----
Mean: -0.0229  -  Median: -0.3002  -  Std: 0.9870
Min: -4.7006  -  25%: -0.5719  -  50%: -0.3002  -  75%: 0.1762  -  Max: 80.4406
Skew: 4.5931  -  Kurtosis: 114.0333
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_561.png)

```
f_281
-----
Mean: 0.0006  -  Median: -0.0310  -  Std: 1.1009
Min: -10.0447  -  25%: -0.6355  -  50%: -0.0310  -  75%: 0.6425  -  Max: 9.8529
Skew: 0.0541  -  Kurtosis: 1.0872
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_563.png)

```
f_282
-----
Mean: -0.0058  -  Median: -0.0748  -  Std: 1.1081
Min: -6.6587  -  25%: -0.8334  -  50%: -0.0748  -  75%: 0.8207  -  Max: 7.1494
Skew: 0.1277  -  Kurtosis: -0.2466
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_565.png)

```
f_283
-----
Mean: 0.0010  -  Median: 0.0014  -  Std: 1.0078
Min: -3.1452  -  25%: -0.8549  -  50%: 0.0014  -  75%: 0.8572  -  Max: 3.4504
Skew: -0.0005  -  Kurtosis: -1.1068
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_567.png)

```
f_284
-----
Mean: -0.0107  -  Median: -0.1828  -  Std: 1.1014
Min: -6.9575  -  25%: -0.8691  -  50%: -0.1828  -  75%: 0.7073  -  Max: 7.4868
Skew: 0.5992  -  Kurtosis: 0.2683
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_569.png)

```
f_285
-----
Mean: 0.0080  -  Median: -0.0755  -  Std: 1.0423
Min: -4.8304  -  25%: -0.8023  -  50%: -0.0755  -  75%: 0.7246  -  Max: 7.3521
Skew: 0.3943  -  Kurtosis: -0.2523
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_571.png)

```
f_286
-----
Mean: -0.0176  -  Median: -0.2157  -  Std: 0.9879
Min: -25.0444  -  25%: -0.3511  -  50%: -0.2157  -  75%: -0.0098  -  Max: 55.5523
Skew: 9.2487  -  Kurtosis: 198.5892
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_573.png)

```
f_287
-----
Mean: -0.0122  -  Median: -0.3500  -  Std: 1.0179
Min: -4.6153  -  25%: -0.8647  -  50%: -0.3500  -  75%: 0.7411  -  Max: 5.4064
Skew: 0.7398  -  Kurtosis: -0.5288
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_575.png)

```
f_288
-----
Mean: -0.0085  -  Median: -0.2895  -  Std: 0.9931
Min: -3.9234  -  25%: -0.4859  -  50%: -0.2895  -  75%: 0.1003  -  Max: 41.0663
Skew: 6.4997  -  Kurtosis: 93.0650
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_577.png)

```
f_289
-----
Mean: -0.0066  -  Median: -0.1764  -  Std: 1.0001
Min: -49.1403  -  25%: -0.5163  -  50%: -0.1764  -  75%: 0.0742  -  Max: 79.6931
Skew: 11.9505  -  Kurtosis: 513.5670
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_579.png)

```
f_290
-----
Mean: 0.0126  -  Median: 0.0609  -  Std: 1.0845
Min: -9.2715  -  25%: -0.6037  -  50%: 0.0609  -  75%: 0.6882  -  Max: 9.2983
Skew: -0.2933  -  Kurtosis: 1.2870
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_581.png)

```
f_291
-----
Mean: 0.0009  -  Median: -0.2023  -  Std: 1.1144
Min: -6.3600  -  25%: -0.8355  -  50%: -0.2023  -  75%: 0.9036  -  Max: 9.7251
Skew: 0.1273  -  Kurtosis: -0.5441
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_583.png)

```
f_292
-----
Mean: -0.0065  -  Median: -0.1910  -  Std: 1.0702
Min: -8.1837  -  25%: -0.6981  -  50%: -0.1910  -  75%: 0.4841  -  Max: 22.3118
Skew: 1.5116  -  Kurtosis: 5.9613
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_585.png)

```
f_293
-----
Mean: 0.0029  -  Median: 0.2164  -  Std: 1.1012
Min: -23.9948  -  25%: -0.1377  -  50%: 0.2164  -  75%: 0.5089  -  Max: 6.5877
Skew: -2.3005  -  Kurtosis: 10.7824
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_587.png)

```
f_294
-----
Mean: -0.0047  -  Median: 0.2087  -  Std: 1.1445
Min: -14.9727  -  25%: -0.9398  -  50%: 0.2087  -  75%: 0.8616  -  Max: 6.9782
Skew: -0.2014  -  Kurtosis: -0.1352
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_589.png)

```
f_295
-----
Mean: -0.0277  -  Median: -0.2908  -  Std: 0.9593
Min: -9.3490  -  25%: -0.5148  -  50%: -0.2908  -  75%: 0.1276  -  Max: 61.4037
Skew: 4.2937  -  Kurtosis: 38.3984
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_591.png)

```
f_296
-----
Mean: -0.0015  -  Median: 0.0070  -  Std: 1.1405
Min: -7.3666  -  25%: -0.9574  -  50%: 0.0070  -  75%: 0.9504  -  Max: 7.6799
Skew: -0.0898  -  Kurtosis: -0.4446
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_593.png)

```
f_297
-----
Mean: -0.0071  -  Median: -0.1650  -  Std: 1.1081
Min: -7.5794  -  25%: -0.7239  -  50%: -0.1650  -  75%: 0.6098  -  Max: 12.4180
Skew: 0.6783  -  Kurtosis: 1.5134
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_595.png)

```
f_298
-----
Mean: -0.0042  -  Median: 0.0230  -  Std: 1.0513
Min: -6.7073  -  25%: -0.8165  -  50%: 0.0230  -  75%: 0.7985  -  Max: 7.0040
Skew: -0.0878  -  Kurtosis: -0.3276
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_597.png)

```
f_299
-----
Mean: -0.0043  -  Median: -0.2824  -  Std: 1.0082
Min: -10.2826  -  25%: -0.5080  -  50%: -0.2824  -  75%: 0.1411  -  Max: 43.3702
Skew: 6.4460  -  Kurtosis: 99.6554
Missing Values: 0/3141410 (0.0000%)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___32_599.png)

## 7. Time Series

Training set is already in time series format as it is sorted by time_id and investment_id columns. Time series of an investment can be selected by directly indexing the investment_id. However, it might be risky to use time series models for predicting target for couple reasons. First, there are lots missing time IDs in different investments. Second, gap between time IDs is not constant and will be shorter in final private test set. Third, some of the investments have very few samples. Those things makes it hard to utilize time property of data.

In [17]:

```
def visualize_time_series(df, investment_id):
    
    df_investment = df.loc[df['investment_id'] == investment_id]

    fig, ax = plt.subplots(figsize=(24, 8), dpi=100)
    ax.plot(df_investment.set_index('time_id')['target'], label='target', linewidth=3)
    for i in range(300):
        ax.plot(df_investment.set_index('time_id')[f'f_{i}'], alpha=0.05)

    ax.set_xlabel('time_id', fontsize=12.5)
    ax.set_ylabel('Features', fontsize=12.5)
    ax.tick_params(axis='x', labelsize=12.5)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.set_title(f'Investment {investment_id} - Features and Target Along Time Axis', fontsize=15, pad=12.5)
    ax.legend(prop={'size': 15})
    
    plt.show()
```

In [18]:

```
visualize_time_series(df_train, 0)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___35_0.png)

## 8. Feature Sequences

Even though there is no order in features, feature sequences can show some insights about target. Samples with lowest and highest target value are visualized below. Their feature sequences are quite different from each other. Some of the peaks and pits are almost occured in the same places but those occurences don't necessarily suggest that features are sequential. It might be related to their time IDs being really close to each other since features were mostly dependent to time IDs.

In [19]:

```
def visualize_feature_sequence(df, idx):
    
    sample = df.loc[idx, [f'f_{i}' for i in range(300)]].reset_index(drop=True)
    target = df.loc[idx, 'target']
    investment_id = df.loc[idx, 'investment_id']
    time_id = df.loc[idx, 'time_id']

    fig, ax = plt.subplots(figsize=(24, 8), dpi=100)
    ax.plot(sample, linewidth=3)
    
    ax.set_xlabel('Features (f_#)', fontsize=12.5)
    ax.set_ylabel('Values', fontsize=12.5)
    ax.tick_params(axis='x', labelsize=12.5)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.set_title(f'Investment {investment_id} - Time ID {time_id} - Target {target:.6f}', fontsize=15, pad=12.5)
    ax.legend(prop={'size': 15})
    
    plt.show()
```

In [20]:

```
visualize_feature_sequence(df_train, 1621688)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___38_0.png)

In [21]:

```
visualize_feature_sequence(df_train, 1639094)
```

![img](https://www.kaggleusercontent.com/kf/86706885/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FVKZ9Wqt10Sxot-qW7gt_Q.9Az5JmZZ070zgrCJpDi-7fR-5nLtLEY-m33LoTN5mKYnLzdny3lXa_WQTzZGJok7AzF5v_u0oRF51FwLjDA4tmUGA3Ot73CYwdgQgnjJQv8XLZvvUi8DFXmAGDTItBd5IekqCkoBl-ReIfS1QiBjVfT8p4UsdNVox6_O3YXAEOkHw2koGeX8GGbWM2p_5FR6Q_D5Kmr27cd7ORr6vATO8rLcMdVPbulrGFWIeHIrGoYU_-NHg4OTb1vz_Q9MIMFa68vxWk8p5kPKaKofjz_b8Ok6I7dqIvQT7hfKrnJCk7T_EvYYlI9UsGExqaMvjn_pYn0vvG-4M_Oo80ji1-LO3gFTOT9QAFr_NYJ1C_WdscSo6gx0ng1hImjgfCS0PxH8O-54cWcuGHQNtbDwVvc4RRwdN_Sj0611qvfp5FZfnefX3zkgMYg-yTvzdf5LohiijY_W6VkDJ9A-NrQUEDI9WJN2o6iR1LUT4C6bCHOzhN8kUhdl3IHE414vyvQnoiG4gbnWwU8IFq8U2Ai4Nah7pf-y67ubC2x11xxL-mGWVcLHoTiDj87RkzDiT56YAtF9y5m8oRWevmjOuC-f-bttJvvz4cPeLLpSnGQJCtXEW5BKlKvupeSjdJ0-u4fuOkyMtZbLK1W524YVI2ywTtW2ZF_yNOsA4_qOt8v-27Z8moe6CaEdwFjrUzUsWJP5BjrX.kXepkbkhGAHMmQzftw-Jzg/__results___files/__results___39_0.png)