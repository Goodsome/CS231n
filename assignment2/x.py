from cs231n.data_utils import get_CIFAR10_data

data = get_CIFAR10_data()
for k, v in data.items():
    print('%s: ' % k, v.shape)