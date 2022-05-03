# Import plotting modules
import matplotlib.pyplot as plt 
import seaborn as sns

# Set default Seaborn style
sns.set()

# Import pandas package
import pandas as pd 

# Read iris data from csv
iris = pd.read_csv('.\iris.csv')

# Check the read data
iris.head

# Save the sepal length values to
# variable '