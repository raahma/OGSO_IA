# Usage : python predict.py --nb_classes 27 --model path/to/model --img path/to_img

from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2
import copy
import numpy as np
import numpy.matlib
import numpy.linalg
from utils.config_utils import load_yaml, build_record_folder, get_args
import argparse
global module_id_mapper, features, grads

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Path to model')
parser.add_argument('--img', type=str, help='Path to input image')
parser.add_argument('--nb_classes', type=int, help='Number of output classes')


classes_list = [
    "Ferrage_et_accessoires_ANTI_FAUSSE_MANOEUVRE",
    "Ferrage_et_accessoires_Busettes",
    "Ferrage_et_accessoires_Butees",
    "Ferrage_et_accessoires_Chariots",
    "Ferrage_et_accessoires_Charniere",
    "Ferrage_et_accessoires_Compas_limiteur",
    "Ferrage_et_accessoires_Renvois_d'angle",
    "Joints_et_consommables_Equerres_aluminium_moulees",
    "Joints_et_consommables_Joints_a_clipser",
    "Joints_et_consommables_Joints_a_coller",
    "Joints_et_consommables_Joints_a_glisser",
    "Joints_et_consommables_Joints_EPDM",
    "Joints_et_consommables_Joints_PVC_aluminium",
    "Joints_et_consommables_Silicone_pour_vitrage_alu",
    "Joints_et_consommables_Visserie_inox_alu",
    "Poignee_carre_7_mm",
    "Poignee_carre_8_mm",
    "Poignee_cremone",
    "Poignee_cuvette",
    "Poignee_de_tirage",
    "Poignee_pour_Levant_Coulissant",
    "Serrure_Cremone_multipoints",
    "Serrure_Cuvette",
    "Serrure_Gaches",
    "Serrure_Pene_Crochet",
    "Serrure_pour_Porte",
    "Serrure_Tringles",
]

class ImgLoader(object):

    def __init__(self, img_size: int):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((510, 510), Image.BILINEAR),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def load(self, image_path: str):
        ori_img = cv2.imread(image_path)
        assert ori_img.shape[2] == 3, "3(RGB) channels is required."
        img = copy.deepcopy(ori_img)
        img = img[:, :, ::-1] # convert BGR to RGB
        img = Image.fromarray(img)
        img = self.transform(img)
        # center crop
        ori_img = cv2.resize(ori_img, (510, 510))
        pad = (510 - self.img_size) // 2
        ori_img = ori_img[pad:pad+self.img_size, pad:pad+self.img_size]
        return img, ori_img


def forward_hook(module: nn.Module, inp_hs, out_hs):
    global features, module_id_mapper
    layer_id = len(features) + 1
    module_id_mapper[module] = layer_id
    features[layer_id] = {}
    features[layer_id]["in"] = inp_hs
    features[layer_id]["out"] = out_hs


def backward_hook(module: nn.Module, inp_grad, out_grad):
    global grads, module_id_mapper
    layer_id = module_id_mapper[module]
    grads[layer_id] = {}
    grads[layer_id]["in"] = inp_grad
    grads[layer_id]["out"] = out_grad


def build_model(pretrainewd_path: str,
                img_size: int,
                fpn_size: int,
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True,
                use_selection: bool = True,
                use_combiner: bool = True,
                comb_proj_size: int = None):
    from pim_module import PluginMoodel

    backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)

    model = \
        PluginMoodel(backbone=backbone,
                     return_nodes=None,
                     img_size = img_size,
                     use_fpn = use_fpn,
                     fpn_size = fpn_size,
                     proj_type = "Linear",
                     upsample_type = "Conv",
                     use_selection = use_selection,
                     num_classes = num_classes,
                     num_selects = num_selects,
                     use_combiner = use_combiner,
                     comb_proj_size = comb_proj_size)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)

    model.eval()

    model.backbone.layers[0].register_forward_hook(forward_hook)
    model.backbone.layers[0].register_full_backward_hook(backward_hook)
    model.backbone.layers[1].register_forward_hook(forward_hook)
    model.backbone.layers[1].register_full_backward_hook(backward_hook)
    model.backbone.layers[2].register_forward_hook(forward_hook)
    model.backbone.layers[2].register_full_backward_hook(backward_hook)
    model.backbone.layers[3].register_forward_hook(forward_hook)
    model.backbone.layers[3].register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer4.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer4.register_full_backward_hook(backward_hook)

    return model


def cal_backward(out, sum_type: str = "softmax"):
    target_layer_names = ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 'comb_outs']

    sum_out = None
    for name in target_layer_names:

        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]

        tmp_out = torch.softmax(tmp_out, dim=-1)

        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out

    results = []

    with torch.no_grad():
        smax = torch.softmax(sum_out, dim=-1)
        pred_score, pred_cls = torch.max(torch.softmax(sum_out, dim=-1), dim=-1)
        pred_score = pred_score[0]
        pred_cls = pred_cls[0]
        backward_cls = pred_cls

        A = np.transpose(np.matlib.repmat(smax[0], num_classes, 1)) - np.eye(num_classes)
        U, S, V = np.linalg.svd(A, full_matrices=True)
        V = V[num_classes-1,:]
        if V[0] < 0:
            V = -V

        V = np.log(V)
        V = V - min(V)
        V = V / sum(V)

        accur = -np.sort(-V)[0:5]
        order = np.argsort(-V)[0:5].tolist()

        for i in range(5):
            results.append({"class": order[i], "accuracy": accur[i]})

    sum_out[0, backward_cls].backward()

    return results


# Set parameters
module_id_mapper, features, grads = {}, {}, {}

args = parser.parse_args()
data_size = 384
fpn_size = 1536
num_classes = args.nb_classes
num_selects = {'layer1': 256, 'layer2': 128, 'layer3': 64, 'layer4': 32}

model = build_model(pretrainewd_path = args.model,
                    img_size = data_size,
                    fpn_size = fpn_size,
                    num_classes = num_classes,
                    num_selects = num_selects)

def run():

    img_url = args.img

    img_loader = ImgLoader(img_size = data_size)
    img, ori_img = img_loader.load(img_url)

    # Predict the top 5 model names and their probabilities
    img = img.unsqueeze(0)
    out = model(img)
    results = cal_backward(out, sum_type="softmax")

    # The following lines should be replaced with sending back to the client side the information put inside print(...)
    print(classes_list[results[0]["class"]] + " predicted with " + str(round(100*results[0]["accuracy"],2)) + "%")
    print(classes_list[results[1]["class"]] + " predicted with " + str(round(100*results[1]["accuracy"],2)) + "%")
    print(classes_list[results[2]["class"]] + " predicted with " + str(round(100*results[2]["accuracy"],2)) + "%")
    print(classes_list[results[3]["class"]] + " predicted with " + str(round(100*results[3]["accuracy"],2)) + "%")
    print(classes_list[results[4]["class"]] + " predicted with " + str(round(100*results[4]["accuracy"],2)) + "%")

    return {
        "top5_classes": [classes_list[results[0]["class"]], classes_list[results[1]["class"]], classes_list[results[2]["class"]], classes_list[results[3]["class"]], classes_list[results[4]["class"]]],
        "top5_probs": [str(round(100*results[0]["accuracy"],2)) + "%", str(round(100*results[1]["accuracy"],2)) + "%", str(round(100*results[2]["accuracy"],2)) + "%", str(round(100*results[3]["accuracy"],2)) + "%", str(round(100*results[4]["accuracy"],2)) + "%"]
    }

if __name__ == '__main__':
    run()
