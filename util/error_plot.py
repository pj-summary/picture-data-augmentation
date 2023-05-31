import matplotlib.pyplot as plt
import csv
import numpy as np

models = ['resnet18']
methods = ['baseline', 'baseline+', 'cutout', 'mixup', 'cutmix']

for model in models:
    method = [[] for i in range(6)]
    with open('../results/'+model+'.csv', 'r') as f:
        f_csv = csv.reader(f)
        k = 0
        for row in f_csv:
            k += 1
            if k == 1:
                continue
            for i in range(6):
                method[i].append(float(row[i]))
    method = np.array(method)
    for i in range(5):
        method[i+1] = 100 - 100*method[i+1]
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Error(%)')
    plt.title(model)
    for i in range(5):
        plt.plot(method[0], method[i+1], label=methods[i])
    plt.legend(methods)
    plt.savefig('../results/'+model+'_error.png', format='png')
    plt.show()
