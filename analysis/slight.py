import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import torch
import random
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def draw_heat_map(df, mask_data, rx_tick, sz_tick, sz_tick_num, rx_tick_num, x_label, z_label, map_title):
    # 用于画图
    c_map = sns.cubehelix_palette(start=1.6, light=0.8, as_cmap=True, reverse=True)
    plt.subplots(figsize=(6, 6))
    # ax = sns.heatmap(df, vmax=600, vmin=500, mask=mask_data, cmap=c_map,
    #                  square=True, linewidths=0.005, xticklabels=rx_tick, yticklabels=sz_tick)

    ax = sns.heatmap(df,mask=mask_data,square=True, linewidths=0.005, xticklabels=rx_tick, yticklabels=sz_tick)

    # ax = sns.heatmap(df,mask=mask_data,
    #                  square=True, linewidths=0.005)

    # ax.set_xticks(rx_tick_num)
    # ax.set_yticks(sz_tick_num)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(z_label)
    # ax.set_title(map_title)
    plt.savefig(map_title + '.pdf', dpi=300)
    plt.show()
    plt.close()

model_outputs_attentions=torch.load("./anllmep.attnm.apple.pt")["attn_matrix"]

i="avg"
fo = open('./appleanllmep/layers.{}.out'.format(i), 'w')

attm = torch.mean(model_outputs_attentions[0], dim=0)



# tgt2srcattm = attm[85:, 56:79]
# tgt2srcattm = attm[12:, 1:12]
tgt2srcattm = attm[14:, 1:14]
print(model_outputs_attentions.size())

# tgten="The ▁Minister ▁of ▁Justice ▁Georg ▁Eisen reich ▁arrived ▁in ▁Fran con ia ▁from ▁the ▁depth s ▁of ▁the ▁sea ▁with ▁his ▁wife ▁An ja .".split()
# srcde="Aus ▁den ▁T ief en ▁des ▁Me eres ▁nach ▁Fran ken ▁kam ▁Just iz minister ▁Georg ▁Eisen reich ▁samt ▁Frau ▁An ja .".split()

# <s> ▁Donald ▁was ▁very ▁good ▁at ▁playing ▁the ▁viol in ▁but ▁Matthew ▁was ▁not . ▁Matthew ▁gave ▁a ▁st unning ▁concert ▁performance .

srcde="▁Donald ▁was ▁very ▁good ▁at ▁playing ▁the ▁viol in ▁but ▁Matthew ▁was ▁not .".split()
tgten="▁Matthew ▁gave ▁a ▁st unning ▁concert ▁performance .".split()

prefix="<s> ▁Donald ▁was ▁very ▁good ▁at ▁playing ▁the ▁viol in ▁but ▁Matthew ▁was ▁not .".split()
# tgten=": T od ay , ▁I ' m ▁going ▁to ▁the ▁park .".split()
#

srcde="▁Apple ▁is ▁del icious . ▁He ▁go ▁to ▁the ▁market . ▁He ▁bu".split()
tgten="ys ▁an ▁apple .".split()



draw_heat_map(tgt2srcattm.detach().to(torch.float).cpu().numpy(), None, srcde, tgten, 0,0,0,0,'./appleanllmep/avg')
print(tgt2srcattm.size(), file=fo, flush=True)
t2sscores=tgt2srcattm.detach().to(torch.float).cpu().numpy()
indexes = t2sscores.argmax(axis=1)
print(indexes, file=fo, flush=True)
for i,x in enumerate(indexes):
    print(tgten[i], srcde[x], file=fo, flush=True)
print(len(indexes), file=fo, flush=True)

# break
# print(attm)
# print(attm.detach().to(torch.float).cpu().numpy())
# plt.figure()
# plot = sns.heatmap(tgt2srcattm.detach().to(torch.float).cpu().numpy())
# plt.savefig("./tgt2srcattn.{}.jpg".format(i))
# plt.close()

# print(tgt2srcattm.size(), file=fo, flush=True)
# t2sscores=tgt2srcattm.detach().to(torch.float).cpu().numpy()
# indexes = t2sscores.argmax(axis=1)
# print(indexes, file=fo, flush=True)
# print(len(indexes), file=fo, flush=True)

layers = [i for i in range(1)]
for i in layers:

    fo = open('./appleanllmep/layers.{}.out'.format(i), 'w')

    attm = model_outputs_attentions[0][i]



    tgt2srcattm = attm[14:, 1:14]
    # tgt2srcattm = attm[12:, 1:12]
    print(model_outputs_attentions.size())
    # break
    # print(attm)
    # print(attm.detach().to(torch.float).cpu().numpy())
    # plt.figure()
    # plot = sns.heatmap(tgt2srcattm.detach().to(torch.float).cpu().numpy())
    # plt.savefig("./tgt2srcattn.{}.jpg".format(i))
    # plt.close()
    # if i==0 or i==31:
    draw_heat_map(tgt2srcattm.detach().to(torch.float).cpu().numpy(), None, srcde, tgten, 0,0,0,0,'./appleanllmep/{}'.format(i))


    print(tgt2srcattm.size(), file=fo, flush=True)
    t2sscores=tgt2srcattm.detach().to(torch.float).cpu().numpy()
    indexes = t2sscores.argmax(axis=1)
    print(indexes, file=fo, flush=True)
    for i,x in enumerate(indexes):
        print(tgten[i], srcde[x], file=fo, flush=True)
    print(len(indexes), file=fo, flush=True)


# layers = [i for i in range(1)]
# for i in layers:

#     fo = open('./senttoday/layers.{}.out'.format(i), 'w')

#     attm = model_outputs_attentions[0][i]



#     tgt2srcattm = attm[:, :]
#     print(model_outputs_attentions.size())
#     # break
#     # print(attm)
#     # print(attm.detach().to(torch.float).cpu().numpy())
#     # plt.figure()
#     # plot = sns.heatmap(tgt2srcattm.detach().to(torch.float).cpu().numpy())
#     # plt.savefig("./tgt2srcattn.{}.jpg".format(i))
#     # plt.close()
#     # if i==0 or i==31:
#     draw_heat_map(tgt2srcattm.detach().to(torch.float).cpu().numpy(), None, srcde, tgten, 0,0,0,0,'./senttoday/{}'.format(i))


#     print(tgt2srcattm.size(), file=fo, flush=True)
#     t2sscores=tgt2srcattm.detach().to(torch.float).cpu().numpy()
#     indexes = t2sscores.argmax(axis=1)
#     print(indexes, file=fo, flush=True)
#     for i,x in enumerate(indexes):
#         print(i,x, file=fo, flush=True)
#     print(len(indexes), file=fo, flush=True)