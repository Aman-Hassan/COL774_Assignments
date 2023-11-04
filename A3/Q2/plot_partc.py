import matplotlib.pyplot as plt

x = [1,2,3,4]
train_acc = [0.56,0.58,0.61,0.62]
test_acc = [0.562,0.57,0.60,0.61]
plt.plot(x,train_acc,label="Train")
plt.plot(x,test_acc,label="Test")
plt.xlabel("Number of hidden layers")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of hidden layers")
plt.legend()
plt.savefig("./Graphs/part_f.png")
plt.show()