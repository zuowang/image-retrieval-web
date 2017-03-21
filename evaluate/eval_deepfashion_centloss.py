import cPickle
input = open('../feat_deepfashion_centloss.pkl', 'rb')
feat = cPickle.load(input)
imNamelist = cPickle.load(input)
input.close()

item_to_imind = {}
for i, im in enumerate(imNamelist):
    item = im.split('/')[-2]
    if item not in item_to_imind:
        item_to_imind[item] = []
    item_to_imind[item].append(i)

import numpy as np
imNum = len(imNamelist)
score = np.zeros(shape=(imNum, 1))
mAP = 0.0
for i, query in enumerate(feat):
    score = np.dot(query, feat.T)
    rankids = np.argsort(score)[::-1]
    query = np.vstack((query, feat[rankids[:10]])).mean(axis=0)
    score = np.dot(query, feat.T)
    rankids = np.argsort(score)[::-1]
    rankids_to_ind = dict(zip(rankids, xrange(len(rankids))))
    item = imNamelist[i].split('/')[-2]
    ap = 0.0
    hit = 0
    for imind in item_to_imind[item]:
        hit += 1
        ap += hit / (rankids_to_ind[imind] + 1)
    ap /= hit
    mAP += ap
mAP /= imNum
print mAP

