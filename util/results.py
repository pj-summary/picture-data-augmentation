import csv
import os
import math

models = ['resnet18']
methods = ['baseline', 'baseline+', 'cutout', 'mixup', 'cutmix']
l = [['0' for k in methods] for j in models]

for model in models:
    m = []
    for method in methods:
        name = './runs/' + '_'.join([model, method]) + '.csv'
        if not os.path.exists(name):
            continue
        k = 0
        b = []
        with open(name, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                try:
                    x = int(row[0])
                    if x == 1:
                        a = []
                    a.append(row[2])
                    if x == 200 and len(a) == 200:
                        b.append(a)
                except:
                    pass
        if len(b) == 1:
            b = [b[0], b[0], b[0], b[0], b[0]]
        m.append(b[0])
        name2 = '../results/' + '_'.join([model, method]) + '.csv'
        with open(name2, 'w') as f:
            f_csv = csv.writer(f, dialect='unix')
            line1 = ['Epoch']
            for i in range(5):
                line1.append('test_acc' + str(i + 1))
            f_csv.writerow(line1)
            for i in range(200):
                f_csv.writerow([i + 1, b[0][i], b[1][i], b[2][i], b[3][i], b[4][i]])
        c = []
        for i in range(5):
            c.append(float(b[i][199])*100)
        miu = sum(c) / 5
        for i in range(5):
            c[i] = (c[i] - miu) * (c[i] - miu)
        s = math.sqrt((sum(c)) / 4)
        bias = 1.96 * s
        l[models.index(model)][methods.index(method)] = '%.2f' % miu + '±' + '%.2f' % bias
    name3 = '../results/' + '_'.join([model]) + '_final.csv'
    with open(name3, 'w') as f:
        f_csv = csv.writer(f, dialect='unix')
        line1 = ['Epoch']
        f_csv.writerow(['Epoch', 'base', 'base+', 'cutout', 'mixup', 'cutmix'])
        for i in range(200):
            f_csv.writerow([i + 1, m[0][i], m[1][i], m[2][i], m[3][i], m[4][i]])

# Confidence intervals
with open('../results/CI.csv', 'w') as f:
    f_csv = csv.writer(f, dialect='unix')
    f_csv.writerow(['model', 'base', 'base+', 'cutout', 'mixup', 'cutmix'])
    for model in models:
        f_csv.writerow([model, *l[models.index(model)]])
