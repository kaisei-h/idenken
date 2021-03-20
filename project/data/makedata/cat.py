import pickle
import torch

#input_test1 = pickle.load(open("40base10M/input_test.pkl","rb"))
#input_test2 = pickle.load(open("40base10M_2/input_test.pkl","rb"))
input_train1 = pickle.load(open("3M/input_train.pkl","rb"))
input_train2 = pickle.load(open("7M/input_train.pkl","rb"))
#target_test1 = pickle.load(open("40base10M/target_test.pkl","rb"))
#target_test2 = pickle.load(open("40base10M_2/target_test.pkl","rb"))
target_train1 = pickle.load(open("3M/target_train.pkl","rb"))
target_train2 = pickle.load(open("7M/target_train.pkl","rb"))

#input_test = torch.cat([input_test1, input_test2])
input_train = torch.cat([input_train1, input_train2])
#target_test = torch.cat([target_test1, target_test2])
target_train = torch.cat([target_train1, target_train2])

#pickle.dump(input_test, open(f"input_test.pkl", "wb"))
pickle.dump(input_train, open(f"input_train.pkl", "wb"))
#pickle.dump(target_test, open(f"target_test.pkl", "wb"))
pickle.dump(target_train, open(f"target_train.pkl", "wb"))
