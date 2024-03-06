import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
print(data.mean(numeric_only=True))
print()
print("Дисперсии:")
print(data.var(numeric_only=True))
print()
print("Корреляции:")
print(data.corr(numeric_only=True))
print()
print("Минимумы:")
print(data.min(numeric_only=True))
print()
print("Максимумы:")
print(data.max(numeric_only=True))
print()
print("Квартили:")
print(data.quantile([0.25, 0.5, 0.75], numeric_only=True))

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
