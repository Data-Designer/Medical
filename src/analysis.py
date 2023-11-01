#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 9:20
# @Author  : Anonymous
# @Site    : 
# @File    : analysis.py
# @Software: PyCharm

# #Desc: 所有曲线
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
from loader_scm import get_meta_split_data_loaders
from sklearn.manifold import TSNE


def click_white():
    file_names = os.listdir('./img/')
    path = r"E:/Code/Pycharm/Medical/draw/img/"
    print(file_names)
    for file_name in file_names:
        os.system('pdfcrop "{}" "{}" '.format(path+file_name,path+file_name))
    print("end!")



def tsne_plot(vectors,labels,title, save_name=None):
    tsne = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)
    fontdict = {'family': 'Times New Roman',
                'style': 'italic',
                'color': 'black',
                'weight': 'normal',
                'size': 17}
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax1.set(yticklabels=[])  # remove the tick labels
    ax1.set(xticklabels=[])  # remove the tick labels


    scatter = ax1.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='viridis')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    ax1.add_artist(legend1)

    plt.title(title, fontdict=fontdict)
    # plt.show()
    plt.savefig("../draw/img/{}.pdf".format(save_name), dpi=600, format='pdf')


def analysis_prompt(model, image_loader, test=False, device='cpu'):
    """tsne"""
    model.eval()
    normal_feas = []
    global_feas = []
    local_feas = []
    with torch.no_grad():
        for img, mask, domain_labels in image_loader:
            img, mask = img.to(device), mask.to(device)
            img_raw = torch.cat((img, model.global_prompt(img, meta_loss=None,
                                                             meta_step_size=None,
                                                             stop_gradient=None)), dim=1)
            img_normal = img.clone().repeat(1,3,1,1)
            encoder_out = model.vision_encoder(img_normal, '0')
            global_encoder_out = model.vision_encoder(img_raw, '0')
            if not test:
                domain_prompt = model.prompt_bank[domain_labels.nonzero(as_tuple=False)[:, 1]]
            else:
                domain_prompt,_ = model.prompt_forward(global_encoder_out[0], model.prompt_bank, None, None, None)
            src_normal = encoder_out[0]
            src_no_spec = global_encoder_out[0]
            src_spec = model.decoder.cross_add_local_prompt(global_encoder_out[0], domain_prompt)  # B, 512, 14, 1
            # seg result
            seg_mask_normal, z_out_normal, _, _ = model.decoder.seg_head(src_normal, encoder_out[1], encoder_out[2], encoder_out[3], img_normal, meta_loss=None,
                                                        meta_step_size=None,
                                                        stop_gradient=None)

            seg_mask_spec, z_out_spec, _, _ = model.decoder.seg_head(src_spec, global_encoder_out[1], global_encoder_out[2], global_encoder_out[3], img_raw, meta_loss=None,
                                                        meta_step_size=None,
                                                        stop_gradient=None)
            seg_mask_no_spec, z_out_no_spec, _, _ = model.decoder.seg_head(src_no_spec, global_encoder_out[1], global_encoder_out[2], global_encoder_out[3], img_raw, meta_loss=None,
                                                        meta_step_size=None,
                                                        stop_gradient=None)
            normal_feas.append(src_normal.reshape(src_normal.shape[0],-1).cpu().numpy())
            global_feas.append(src_no_spec.reshape(src_no_spec.shape[0],-1).cpu().numpy())
            local_feas.append(src_spec.reshape(src_spec.shape[0],-1).cpu().numpy())
            # normal_feas.append(z_out_normal.cpu().numpy())
            # global_feas.append(z_out_no_spec.cpu().numpy())
            # local_feas.append(z_out_spec.cpu().numpy())
    normal_feas, global_feas, local_feas = np.concatenate(normal_feas), np.concatenate(global_feas), np.concatenate(local_feas)

    print("run success!")
    return normal_feas, global_feas, local_feas




if __name__ == '__main__':
    # tsne
    device ='cuda:1'
    root = '/home/xxx/Medical/log/ckps'
    model_name = 'promptransunet'
    dataset = 'SCM'
    batch_sz = 1
    test_vendor = 'A'
    ratio = '1'  # 0.02
    pth = 'x.pth'
    aba = ''
    model_path = os.path.join(root, model_name + '-' + dataset + '-' + test_vendor + '-' + ratio + aba, pth)


    model = torch.load(model_path)
    model.to(device)
    print('model load success!')
    test_vendor = 'A'
    domain_1_loader, _, \
    domain_2_loader, _, \
    domain_3_loader, _, \
    test_loader, \
    _, _, _ = get_meta_split_data_loaders(4, test_vendor=test_vendor, image_size=288)
    print('data load success!')

    normal_feas_1, global_feas_1, local_feas_1 = analysis_prompt(model, domain_1_loader, test=False, device=device)
    normal_feas_2, global_feas_2, local_feas_2 = analysis_prompt(model, domain_2_loader, test=False, device=device)
    normal_feas_3, global_feas_3, local_feas_3 = analysis_prompt(model, domain_3_loader, test=False, device=device)
    normal_feas_t, global_feas_t, local_feas_t = analysis_prompt(model, test_loader, test=True, device=device)

    label1, label2, label3, label_t = np.zeros(normal_feas_1.shape[0]), np.zeros(normal_feas_2.shape[0])+1, \
                                      np.zeros(normal_feas_3.shape[0])+2, np.zeros(normal_feas_t.shape[0])+3

    # print(normal_feas_1.shape, normal_feas_2.shape,normal_feas_3.shape, normal_feas_t.shape)
    normal_feas = np.concatenate([normal_feas_1, normal_feas_2, normal_feas_3, normal_feas_t])
    global_feas = np.concatenate([global_feas_1, global_feas_2, global_feas_3, global_feas_t])
    local_feas = np.concatenate([local_feas_1, local_feas_2, local_feas_3, local_feas_t])
    labels = np.concatenate([label1, label2, label3, label_t])
    tsne_plot(normal_feas, labels, title='Initial Distribution',save_name='initial')
    tsne_plot(global_feas, labels,  title='Add Global Prompt',save_name='global')
    tsne_plot(local_feas, labels, title='Add Local Prompt', save_name='local')




    print("white crop")
    click_white()