from scipy.stats import ttest_ind, ttest_1samp

model1 = [0.569, 0.571]
model2 = [0.583, 0.566]
model3 = [0.585, 0.592]

baseline = 0.578

if __name__ == '__main__':
    t, p = ttest_ind(model1, model2)
    print(f'model1 vs model2: p={p}, t={t}')

    t, p = ttest_ind(model1, model3)
    print(f'model1 vs model3: p={p}, t={t}')

    t, p = ttest_ind(model2, model3)
    print(f'model2 vs model3: p={p}, t={t}')

    t, p = ttest_1samp(model1, baseline)
    print(f'model1 vs baseline: p={p}, t={t}')
