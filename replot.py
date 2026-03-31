import pickle
import matplotlib.pyplot as plt

with open("results/experiment_raw_data.pkl", "rb") as f:
    data = pickle.load(f)

# Now you have access to all your lists instantly!
# example: data["exp1"]["test"] contains all the K-dictionaries