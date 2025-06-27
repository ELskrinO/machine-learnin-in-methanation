import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_graphviz

# 1. Прочитать файл
df = pd.read_excel('Metan.xlsx')

# 2. Формируем X и y
X = df.iloc[:, :-3]
y = df.iloc[:, -1]

# 3. Обучаем регрессионное дерево
reg = DecisionTreeRegressor(max_depth=3, random_state=42)
reg.fit(X, y)

# 4. Экспорт дерева в файл .dot для GraphvizOnline
export_graphviz(
    reg,
    out_file='tree.dot',
    feature_names=X.columns,
    filled=True,
    rounded=True,
    special_characters=True
)
