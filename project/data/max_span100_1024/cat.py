import pickle
import torch

input_val0 = pickle.load(open("input_val0.pkl","rb"))
target_val0 = pickle.load(open("target_val0.pkl","rb"))
input_train0 = pickle.load(open("input_train0.pkl","rb"))
target_train0 = pickle.load(open("target_train0.pkl","rb"))
input_val1 = pickle.load(open("input_val1.pkl","rb"))
target_val1 = pickle.load(open("target_val1.pkl","rb"))
input_train1 = pickle.load(open("input_train1.pkl","rb"))
target_train1 = pickle.load(open("target_train1.pkl","rb"))


input_val = torch.cat([input_val0, input_val1])
input_train = torch.cat([input_train0, input_train1])
target_val = torch.cat([target_val0, target_val1])
target_train = torch.cat([target_train0, target_train1])
print(input_val.shape)
print(target_train.shape)

pickle.dump(input_val, open(f"input_val01.pkl", "wb"))
pickle.dump(input_train, open(f"input_train01.pkl", "wb"))
pickle.dump(target_val, open(f"target_val01.pkl", "wb"))
pickle.dump(target_train, open(f"target_train01.pkl", "wb"))
