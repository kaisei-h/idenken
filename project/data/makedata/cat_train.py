import pickle
import torch


input_1 = pickle.load(open("input_train1.pkl","rb"))
target_1 = pickle.load(open("target_train1.pkl","rb"))
print(target_1.shape)
input_2 = pickle.load(open("input_train2.pkl","rb"))
target_2 = pickle.load(open("target_train2.pkl","rb"))
print(target_2.shape)
input_3 = pickle.load(open("input_train3.pkl","rb"))
target_3 = pickle.load(open("target_train3.pkl","rb"))
print(target_3.shape)
input_4 = pickle.load(open("input_train4.pkl","rb"))
target_4 = pickle.load(open("target_train4.pkl","rb"))
print(target_4.shape)
input_5 = pickle.load(open("input_train5.pkl","rb"))
target_5 = pickle.load(open("target_train5.pkl","rb"))
print(target_5.shape)


input_train = torch.cat([input_1, input_2, input_3, input_4, input_5])
target_train = torch.cat([target_1, target_2, target_3, target_4, target_5])
print(input_train.shape)
print(target_train.shape)

pickle.dump(input_train, open(f"input_train.pkl", "wb"))
pickle.dump(target_train, open(f"target_train.pkl", "wb"))