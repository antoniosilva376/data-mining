
import csv
import numpy as np

class Dataset:
    #use y= anything for it to be created
    def __init__(self, X=None, y=None, column_names=None):
        self.X = X
        self.y = y
        self.column_names = column_names
        
    def set_X(self, X):
        self.X = X
        
    def set_y(self, y):
        self.y = y
        
    def set_column_names(self, column_names):
        self.column_names = column_names
        
    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def get_column_names(self):
        return self.column_names
            
    def read_csv(self, filename, sep=','):
        with open(filename, 'r') as f:
            lines = f.readlines()
            data = [line.strip().split(',') for line in lines[1:]]
        if self.y is None:
            self.column_names = lines[0].strip().split(',')
            self.X = np.array(data)
        else:
            self.column_names = lines[0].strip().split(',')[:-1]
            self.X = np.array(data)[:, :-1]
            self.y = np.array(data)[:, -1]
        
    def count_nulls(self):
        return np.sum(self.X == '', axis=0)
          
    def fill_nulls(self):
        pass

    def describe(self):
        pass



ds = Dataset()
ds.read_csv('notas.csv')
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))
print("column_names:\n" + str(ds.column_names))
print("count_nulls:\n" + str(ds.count_nulls()))
ds.fill_nulls()
print("-------------------")
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))
print("column_names:\n" + str(ds.column_names))
print("count_nulls:\n" + str(ds.count_nulls()))
ds.describe()

