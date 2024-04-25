import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Вычисляем выбросы с помощью межквартильного размаха
def outliers(column, dataset):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = dataset[(dataset[column] < lower_bound) | (dataset[column] > upper_bound)]
    dataset = dataset[(dataset[column] > lower_bound) & (dataset[column] < upper_bound) & (dataset[column] > 0)]

    print("Выбросы в столбце '{}':".format(column))
    print(outliers[column])

    return dataset


data = pd.read_csv("hypothyroid.csv")
data_columns = data.columns.difference(["Age", "sex", "TSH", "T3", "T4", "T4U", "FTI", "TBG", "referral source",
                                        "binaryClass"])
data[data_columns] = data[data_columns].map(lambda x: False if (x == 'f') else True)

# Столбцы
columns = ["Age", "TSH", "T3", "T4", "T4U", "FTI"]

print("---Основная информация о данных---")
print()
print("Количество наблюдений: {}".format(data.shape[0]))
print("Количество перемнных: {}".format(data.shape[1]))
print()
print("Типы данных:")
print(data.dtypes)
print()
print("Количество пропущенных значений:")
print(data.isnull().sum())
print()

# Удаляем строки с пропущенными значениями
data = data.dropna()

# Находим и удаляем выбросы
for column in columns:
    data = outliers(column, data)
    print()

print("---Основные статистические характеристики данных---")
print()
print("Средние значения:")
print(data.loc[:, columns].mean(numeric_only=True))
print()
print("Дисперсии:")
print(data.loc[:, columns].var(numeric_only=True))
print()
print("Корреляции:")
print(data.loc[:, columns].corr(numeric_only=True))
print()
print("Минимумы:")
print(data.loc[:, columns].min(numeric_only=True))
print()
print("Максимумы:")
print(data.loc[:, columns].max(numeric_only=True))
print()
print("Квартили:")
print(data.loc[:, columns].quantile([0.25, 0.5, 0.75], numeric_only=True))
print()

# Гистрограммы с графиком функции распределения
for column in columns:
    sns.histplot(data, x=column, kde=True)
    plt.show()

# Боксплоты
for column in columns:
    sns.boxplot(data, y=column)
    plt.show()

# Диаграммы рассеивания
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        sns.scatterplot(data=data, x=columns[i], y=columns[j])
        plt.show()

# Разделение данных на две группы
hyper_data = data[data['queryhyperthyroid'] is True]
nohyper_data = data[data['queryhyperthyroid'] is False]

# Проведение t-теста
t_test, p_value = stats.ttest_ind(hyper_data['TSH'], nohyper_data['TSH'], equal_var=False)

# Вывод результатов
print("t-тест:", t_test)
print("p-value:", p_value)

# Оценка значимости различий
alpha = 0.05
if p_value < alpha:
    print("Различия статистически значимы")
else:
    print("Нет статистически значимых различий")
