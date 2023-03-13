import numpy as np

class Dataset:
    def __init__(self, X=None, y=None, features=None, label=None):
        self.X = X
        self.y = y
        self.features = features
        self.label = label
        
    def set_X(self, X):
        self.X = X
        
    def set_y(self, y):
        self.y = y
        
    def set_features(self, features):
        self.features = features

    def set_label(self, label):
        self.label=label
        
    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def get_features(self):
        return self.features

    def get_label(self):
        return self.label
            
    def read_csv(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                data = [line.strip().split(',') for line in lines[1:]]
                self.features = lines[0].strip().split(',')
            if self.y is None:
                self.X = np.array(data)
            else:
                self.X = np.array(data)[:, :-1]
                self.y = np.array(data)[:, -1]
        except FileNotFoundError:
            print(f'file not found')

    def write_csv(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write(','.join(self.features_names) + ',' + self.label + '\n')
                for row in range(self.X.shape[0]):
                    f.write(','.join([str(elem) for elem in self.X[row]]) + ',' + str(self.y[row]) + '\n')
        except IOError:
            print(f'error writing csv')

    def read_tsv(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                data = [line.strip().split('\t') for line in lines[1:]]
                self.features = lines[0].strip().split('\t')
            if self.y is None:
                self.X = np.array(data)
            else:
                self.X = np.array(data)[:, :-1]
                self.y = np.array(data)[:, -1]
        except FileNotFoundError:
            print(f'file not found')

    def write_tsv(self, filename):
        try:
            with open(filename, 'w') as f:
                header = '\t'.join(['col' + str(i) for i in range(self.X.shape[1])])
                header += '\tlabel\n'
                f.write(header)
                for row in range(self.X.shape[0]):
                    row_values = '\t'.join([str(elem) for elem in self.X[row]])
                    row_values += '\t' + str(self.y[row]) + '\n'
                    f.write(row_values)
        except IOError:
            print(f"error writing tsv")

        
    def count_nulls(self):
        return np.sum(self.X == '', axis=0)
          
    #needs median version
    def fill_nulls(self):
        X2 = self.X[:, 1:]
        X2[X2 == ''] = np.nan
        means = np.nanmean(X2.astype(float), axis=1)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if self.X[i][j] == 'nan':
                    self.X[i][j] = means[i]

    def describe(self):
        num_elements = self.X.shape[0]
        means = np.mean(self.X[:, 1:].astype(float), axis=0)
        stds = np.std(self.X[:, 1:].astype(float), axis=0)
        mins = np.min(self.X[:, 1:].astype(float), axis=0)
        maxs = np.max(self.X[:, 1:].astype(float), axis=0)
        q25 = np.percentile(self.X[:, 1:].astype(float), 25, axis=0)
        q50 = np.percentile(self.X[:, 1:].astype(float), 50, axis=0)
        q75 = np.percentile(self.X[:, 1:].astype(float), 75, axis=0)

        for i, feature_name in enumerate(self.features[1:]):
            if feature_name != "":
                print("--------------------------------\n")
                print("Feature:", feature_name)
                print("Number of elements:", num_elements)
                print("Minimum value:", mins[i])
                print("Maximum value: ", maxs[i])
                print("Mean:", means[i])
                print("Standard deviation:", stds[i])
                print("25th percentile:", q25[i])
                print("50th percentile (median):", q50[i])
                print("75th percentile:", q75[i])
                print("--------------------------------\n")


"""
ds = Dataset()
ds.read_csv('notas.csv')
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))
print("features:\n" + str(ds.features))
print("count_nulls:\n" + str(ds.count_nulls()))
ds.fill_nulls()
print("-------------------")
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))
print("features:\n" + str(ds.features))
print("count_nulls:\n" + str(ds.count_nulls()))
ds.describe()"""


