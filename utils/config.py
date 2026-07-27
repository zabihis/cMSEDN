import torch as t


# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        #Choose embedding method for node feature
        
        self.k = 4  # mer

        # Parameters of the hidden layer of the graph convolutional networks
        self.hidden_dim = 64
           
        # Number of dataset categories
        self.n_classes = 2

        # Set random seeds
        self.seed = 40  
                
        # Also update device here
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def set_seed(self,s):
        self.seed=s

    def set_d(self,d):
        self.d=d

    def set_n_classes(self,n):
        self.n_classes=n
      
