import io
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("해빙 면적과 해수면의 상승은 어떤 관계가 있을까?")

seaice_file_path = "seaice.csv"
sealevel_file_path = "sea_levels_2015.csv"

df_seaice = pd.read_csv(seaice_file_path)
df_sealevel = pd.read_csv(sealevel_file_path)

st.subheader("Sea Ice Data - Before Cleaning")
st.write(df_seaice.head())

df_seaice.columns = df_seaice.columns.str.strip()

st.subheader("Sea Ice Data - 컬럼 이름 공백 제거")
st.write(df_seaice.head())

# 데이터 정보 출력 (Sea Ice)
st.subheader("Sea Ice Data Info")
seaice_info_buffer = io.StringIO()
df_seaice.info(buf=seaice_info_buffer)
seaice_info = seaice_info_buffer.getvalue()
st.text(seaice_info)

# 데이터 정보 출력 (Sea Level)
st.subheader("Sea Level Data Info")
sealevel_info_buffer = io.StringIO()
df_sealevel.info(buf=sealevel_info_buffer)
sealevel_info = sealevel_info_buffer.getvalue()
st.text(sealevel_info)


# 데이터 미리보기
st.subheader("Sea Ice Data - Preview")
st.dataframe(df_seaice)

st.subheader("Sea Level Data - Preview")
st.dataframe(df_sealevel)

''''
'''


import seaborn as sns
import matplotlib.pyplot as plt

# 산점도 섹션 제목
st.subheader("Scatter Plot: Year vs Extent by Hemisphere")

x_column = 'Year'
y_column = 'Extent'
hue_column = 'hemisphere'

# Matplotlib를 사용해 그래프 생성
fig, ax = plt.subplots(figsize=(12, 8))  # 그래프 크기 설정

sns.stripplot(
    x=x_column,
    y=y_column,
    hue=hue_column,
    data=df_seaice,
    dodge=True,
    palette='Set1',
    alpha=0.3,
    jitter=True,
    ax=ax
)

ax.set_title(f'Scatter plot of {x_column} vs {y_column} by {hue_column}', fontsize=16)
ax.set_xlabel(x_column, fontsize=14)
ax.set_ylabel(y_column, fontsize=14)

# x축 레이블 간격 설정 및 회전
current_values = ax.get_xticks()
ax.set_xticks(current_values[::5])  # 5년 간격으로 표시
ax.tick_params(axis="x", rotation=45)

ax.legend(loc='upper right')
ax.grid(True)

st.pyplot(fig)

df_north_sea_ice_2013 = df_seaice[(df_seaice['hemisphere'] == 'north') & (df_seaice['Year'] <= 2013)]

st.subheader("Filtered Data: North Hemisphere Sea Ice (Up to 2013)")
st.write(df_north_sea_ice_2013.head())


df_north_group_sea_ice_2013 = df_north_sea_ice_2013.groupby('Year')['Extent'].mean().reset_index()

st.subheader("Average Sea Ice Extent per Year (Northern Hemisphere, <= 2013)")
st.write(df_north_group_sea_ice_2013)

st.text("Grouped Data Info:")
group_info_buffer = io.StringIO()
df_north_group_sea_ice_2013.info(buf=group_info_buffer)
group_info = group_info_buffer.getvalue()
st.text(group_info)

st.subheader("Visualization: Average Sea Ice Extent per Year")

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    x='Year',
    y='Extent',
    data=df_north_group_sea_ice_2013,
    s=100,
    color='blue',
    ax=ax
)

ax.set_title('Average Sea Ice Extent per Year (Northern Hemisphere, <= 2013)', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Average Extent', fontsize=14)

ax.tick_params(axis='x', rotation=45)

ax.grid(True)

st.pyplot(fig)



df_sealevel['Year'] = pd.to_datetime(df_sealevel['Time']).dt.year

df_sealevel_group = df_sealevel.groupby('Year')['GMSL'].mean().reset_index()

st.subheader("Average GMSL per Year")
st.write(df_sealevel_group)

import io
sealevel_info_buffer = io.StringIO()
df_sealevel_group.info(buf=sealevel_info_buffer)
sealevel_info = sealevel_info_buffer.getvalue()
st.text("Grouped Data Info:")
st.text(sealevel_info)

st.subheader("Visualization: Average GMSL per Year")

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    x='Year',
    y='GMSL',
    data=df_sealevel_group,
    s=100,
    color='blue',
    ax=ax
)

ax.set_title('Average GMSL per Year', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Average GMSL', fontsize=14)

ax.tick_params(axis='x', rotation=45)

ax.grid(True)

st.pyplot(fig)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.subheader("North Hemisphere Grouped Sea Ice Data (<= 2013)")
st.write(df_north_group_sea_ice_2013)

train_df, test_df = train_test_split(df_north_group_sea_ice_2013, test_size=0.3, random_state=42)

st.subheader("Training and Testing Data Info")
st.text("Training Data Info:")
train_info_buffer = io.StringIO()
train_df.info(buf=train_info_buffer)
st.text(train_info_buffer.getvalue())

st.text("Testing Data Info:")
test_info_buffer = io.StringIO()
test_df.info(buf=test_info_buffer)
st.text(test_info_buffer.getvalue())

X_train = train_df[['Year']]
y_train = train_df['Extent']
X_test = test_df[['Year']]
y_test = test_df['Extent']

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 예측
y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"MSE: {mse}")
st.write(f"RMSE: {rmse}")
st.write(f"MAE: {mae}")
st.write(f"R²: {r2}")

st.subheader("Actual vs Predicted Extent")

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_test, y_test, color='blue', label='Actual')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Extent', fontsize=14)
ax.set_title('Actual vs Predicted Extent', fontsize=16)
ax.legend()
ax.grid(True)

st.pyplot(fig)



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.subheader("Sea Level Grouped Data")
st.write(df_sealevel_group)

train_df, test_df = train_test_split(df_sealevel_group, test_size=0.3, random_state=42)

st.subheader("Training and Testing Data Info")
st.text("Training Data Info:")
train_info_buffer = io.StringIO()
train_df.info(buf=train_info_buffer)
st.text(train_info_buffer.getvalue())

st.text("Testing Data Info:")
test_info_buffer = io.StringIO()
test_df.info(buf=test_info_buffer)
st.text(test_info_buffer.getvalue())

X_train = train_df[['Year']]
y_train = train_df['GMSL']
X_test = test_df[['Year']]
y_test = test_df['GMSL']

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"R²: {r2:.2f}")

st.subheader("Visualization: Actual vs Predicted GMSL")

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_test, y_test, color='blue', label='Actual')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('GMSL', fontsize=14)
ax.set_title('Actual vs Predicted GMSL', fontsize=16)
ax.legend()
ax.grid(True)

st.pyplot(fig)


df_merged = pd.merge(df_north_group_sea_ice_2013, df_sealevel_group, on='Year')

st.subheader("Merged Data: Extent vs GMSL")
st.write(df_merged)

st.subheader("Visualization: Extent vs GMSL with Trend Line")

fig, ax = plt.subplots(figsize=(12, 8))
sns.regplot(
    x='Extent',
    y='GMSL',
    data=df_merged,
    scatter_kws={'s':100},
    line_kws={'color':'black'},
    ci=None,
    ax=ax
)
ax.set_title('Scatter plot of Extent vs GMSL with Trend Line', fontsize=16)
ax.set_xlabel('Extent', fontsize=14)
ax.set_ylabel('GMSL', fontsize=14)
ax.grid(True)

st.pyplot(fig)

correlation = df_merged['Extent'].corr(df_merged['GMSL'])
st.subheader("Correlation Coefficient")
st.write(f"The correlation coefficient between Extent and GMSL is: **{correlation:.2f}**")
