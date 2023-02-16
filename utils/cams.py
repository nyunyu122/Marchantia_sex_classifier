# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.base_cam import BaseCAM

from utils.misc import get_unnormalize

    
class GradCAM_withGrad(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, compute_input_gradient=True):
        super(
            GradCAM_withGrad,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            compute_input_gradient)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))


def calculate_cam(input, targets, cam):
    if cam.compute_input_gradient:
        activations_and_grads = ActivationsAndGradients(cam.model, cam.target_layers,
                                                        reshape_transform=None)
        grayscale_cam = cam(input_tensor=input, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        activations = activations_and_grads.activations[0]
        gradients = activations_and_grads.gradients[0]
        return grayscale_cam, activations, gradients

    elif cam.compute_input_gradient == False:
        grayscale_cam = cam(input_tensor=input, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

def input_to_img(input, unnormalize):
    rgb_img = unnormalize(input[0, :]).detach().cpu().numpy() # get original numpy image via unnormalization
    rgb_img = np.moveaxis(rgb_img, 0, -1)# from channel-first to channel-last    
    rgb_img = np.clip(rgb_img, 0, 1.0)
    return rgb_img


def get_cam_visualizations(input, cam, unnormalize=get_unnormalize()):
    rgb_img = input_to_img(input, unnormalize)
    # grayscale_cams = []
    visualizations = []
    for category in range(2):
        targets = [ClassifierOutputTarget(category) for _ in range(input.shape[0])]
        grayscale_cam = calculate_cam(input, targets, cam) 
        # grayscale_cams.append(grayscale_cam)        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        visualizations.append(visualization)
    return rgb_img, visualizations


def get_cam_visualizations_withGrad(input, cam, unnormalize=get_unnormalize()):
    rgb_img = input_to_img(input, unnormalize)
    visualizations = []
    activations_list, gradients_list = [], []
    for category in range(2):
        targets = [ClassifierOutputTarget(category) for _ in range(input.shape[0])]
        grayscale_cam, activations, gradients = calculate_cam(input, targets, cam) 
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        visualizations.append(visualization)
        activations_list.append(activations)
        gradients_list.append(gradients)    
    return rgb_img, visualizations, activations_list, gradients_list


def plot_cam_results(img, label, pred, visualizations):
    ## plot, ここを関数に
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    ax[0].imshow(img)
    ax[0].set_title("True label: " + str(label) + ", prediction: " + str(pred))
    ax[1].imshow(visualizations[0])
    ax[1].set_title("Grad-CAM for label 0")
    ax[2].imshow(visualizations[1])
    ax[2].set_title("Grad-CAM for label 1")


def plot_gradients_activations(label, pred, gradients_list, activations_list, 
                               save=False, figures_path=None, idx_arglogit=None):
    print("correct label: ", label)
    print('prediction correct: ', (label == pred))

    grads_mean = [np.mean(x.cpu().numpy(), axis=(2, 3)).squeeze() for x in gradients_list]
    activations_mean = [np.mean(x.cpu().numpy(), axis=(2, 3)).squeeze() for x in activations_list]
    products_mean = [x*y for (x, y) in zip(grads_mean, activations_mean)]
    
    for category in range(2):
#         targets = [ClassifierOutputTarget(category) for _ in range(input.shape[0])]  
        print('cam for the correct class: ', (category == label)) 

        fig, ax = plt.subplots(1,6, figsize =(20,3))
        ax = ax.ravel()
        ax[0].hist(grads_mean[category])
        ax[0].set_title('gradients, alpha')
        ax[1].hist(activations_mean[category])
        ax[1].set_title('activations')
        ax[2].hist(products_mean[category])
        ax[2].set_title('Hadamard')
        ax[3].scatter(grads_mean[category], activations_mean[category])
        ax[3].set_title('grads-activations')
        ax[4].scatter(grads_mean[category], products_mean[category])
        ax[4].set_title('grads-products')
        ax[5].scatter(activations_mean[category], products_mean[category])
        ax[5].set_title('activations-products')
        if save == True:
            plt.savefig(f'{figures_path}cam_best{int(idx_arglogit+1)}_label-{label}_preds_{label==pred}_camCorrectClass-{category == label}.png', dpi=300)
            plt.savefig(f'{figures_path}cam_best{int(idx_arglogit+1)}__label-{label}_preds_{label==pred}_camCorrectClass-{category == label}.pdf', dpi=300)
    plt.show()
