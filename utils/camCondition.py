import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

import torch

from utils.cams import get_cam_visualizations, plot_cam_results

class CamConditionContainer(NamedTuple):
    correct: bool
    pred_label: str
    logits_best: bool

        
    def get_conditioned_indices(self, outputs, labels):
        '''correct/incorrect, M/F'''
        preds = np.argmax(outputs, axis=1)
        outputs_sub = outputs[:, 0] - outputs[:,1]
        cond1 = (preds == labels) if self.correct==True else (preds != labels) if self.correct==False else None
        cond2 = (outputs_sub > 0) if self.pred_label == 'F' else (outputs_sub < 0) if self.pred_label == 'M' else None# if self.pred_label ==None
        conditioned_indices = np.where(cond1 & cond2)
        return conditioned_indices

       
    def data_subsets_sorted(self, outputs, labels, inputs):
        '''make conditioned dataset & sort'''
        idx = self.get_conditioned_indices(outputs, labels)
        preds = np.argmax(outputs, axis=1)
        outputs_sub = outputs[:, 0] - outputs[:,1]
        inputs_small, labels_small, outputs_sub_small, outputs_small, preds_small = inputs[idx], labels[idx], outputs_sub[idx], outputs[idx], preds[idx]
        indices_sorted = np.argsort(np.abs(outputs_sub_small))[::-1] if self.logits_best == True else np.argsort(np.abs(outputs_sub_small)) if self.logits_best == False else None
        return indices_sorted, inputs_small, labels_small, outputs_small, outputs_sub_small

        
    def plot_best_heatmaps(self, outputs, labels, inputs, cam, dev, args,
                           N_plot=3, save=True):
        '''plot first N_plot images'''
        indices_sorted, inputs_small, labels_small, outputs_small, outputs_sub_small = self.data_subsets_sorted(outputs, labels, inputs)
        preds_small = np.argmax(outputs_small, axis=1)
        print(self)
        if N_plot <= indices_sorted.shape[0]: 
            for n in range(N_plot):
                i = indices_sorted[n]
                input = inputs_small[i:i+1, :, :, :]
                label = labels_small[i]
                pred = preds_small[i]
                output = outputs_small[i]
                rgb_img, visualizations = get_cam_visualizations(torch.from_numpy(input).clone().to(dev), cam)
                print('output:', output)
                plot_cam_results(rgb_img, label, pred, visualizations)
                if save==True:
                    sorting = 'best' if self.logits_best == True else 'worst'
                    plt.savefig(f'{args.figure_path}heatmap_pred{self.pred_label}_corr{self.correct}_{sorting}{n+1}.png')
                    plt.savefig(f'{args.figure_path}heatmap_pred{self.pred_label}_corr{self.correct}_{sorting}{n+1}.pdf')
                plt.show()
        else:
            print('Not enough samples ', indices_sorted.shape[0])
