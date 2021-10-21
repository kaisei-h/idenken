import pickle
import torch


input_1 = pickle.load(open("input_val1.pkl","rb"))
target_1 = pickle.load(open("target_val1.pkl","rb"))
print(target_1.shape)
input_2 = pickle.load(open("input_val2.pkl","rb"))
target_2 = pickle.load(open("target_val2.pkl","rb"))
print(target_2.shape)
input_3 = pickle.load(open("input_val3.pkl","rb"))
target_3 = pickle.load(open("target_val3.pkl","rb"))
print(target_3.shape)
input_4 = pickle.load(open("input_val4.pkl","rb"))
target_4 = pickle.load(open("target_val4.pkl","rb"))
print(target_4.shape)
input_5 = pickle.load(open("input_val5.pkl","rb"))
target_5 = pickle.load(open("target_val5.pkl","rb"))
print(target_5.shape)


input_val = torch.cat([input_1, input_2, input_3, input_4, input_5])
target_val = torch.cat([target_1, target_2, target_3, target_4, target_5])
print(input_val.shape)
print(target_val.shape)

pickle.dump(input_val, open(f"input_val.pkl", "wb"))
pickle.dump(target_val, open(f"target_val.pkl", "wb"))