import pandas as pd


class Prism:
    def __init__(self):
        self.rules = []

    def fit(self, data, target):
        data = data.copy()
        classes = data[target].unique()
        for c in classes:
            data_c = data[data[target] == c]
            while len(data_c) > 0:
                rule = {}
                max_accuracy = 0
                for col in data.columns[:-1]:
                    values = data_c[col].unique()
                    for val in values:
                        accuracy = len(data_c[(data_c[col] == val) & (data_c[target] == c)]) / len(
                            data_c[data_c[col] == val])
                        if accuracy > max_accuracy:
                            max_accuracy = accuracy
                            rule = {col: val}
                self.rules.append((rule, c))
                rows_to_remove = data_c.index
                for key, value in rule.items():
                    rows_to_remove = rows_to_remove.intersection(data_c[data_c[key] == value].index)
                data_c = data_c.drop(rows_to_remove)

    def predict(self, data):
        predictions = []
        for _, row in data.iterrows():
            for rule, c in self.rules:
                if all(row[key] == value for key, value in rule.items()):
                    predictions.append(c)
                    break
            else:
                predictions.append(None)
        return predictions

    def __repr__(self):
        return '\n'.join(
            [f'IF {" AND ".join([f"{key}={value}" for key, value in rule.items()])} THEN {c}' for rule, c in
             self.rules])


data = pd.DataFrame({'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
                     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
                     'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High','Normal','High'],
                     'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
                     'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']})

prism = Prism()
prism.fit(data, target='PlayTennis')
print(prism)

test_data = pd.DataFrame({'Outlook': ['Overcast'],
                          'Temperature': ['Hot'],
                          'Humidity': ['High'],
                          'Wind': ['Weak']})
print(prism.predict(test_data))
