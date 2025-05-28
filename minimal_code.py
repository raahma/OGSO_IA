import torch
import warnings
warnings.simplefilter("ignore")
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import numpy.matlib
import numpy.linalg
import cv2
import timm
from PIL import Image
import copy

global module_id_mapper
global features
global grads


#weapons_list = ['1853+Enfield+rifle-musket', 'AEK-919K+Kashtan+submachine+gun', 'AK+101', 'AK+47', 'AK+74', 'AK-12+rifle', 'AKS+74U', 'AN-94+rifle', 'AS+Val+rifle', 'Accuracy+International+Arctic+Warfare+Magnum', 'ArmaLite+AR10', 'ArmaLite+AR15', 'Astra+600', 'Astra+Modele+900', 'Atchisson+AA-12+shotgun', 'Barrett+M107', 'Barrett+M95', 'Bauer+Automatic+25', 'Benelli+M4+Super+90', 'Berdan+Sharps+Rifle', 'Beretta+92FS', 'Beretta+98FS', 'Beretta+ARX100+rifle', 'Beretta+M9A1+pistol', 'Bren+light+machine+gun', 'Brown+Bess+musket', 'Browning+BT-99+Shotgun', 'Browning+Hi-Power+Mark+III+pistol', 'Bushmaster+AR+15+Semiautomatic+Rifle', 'COLT+WOODSMAN+pistol', 'CZ+75+Tactical+Sports+pistol', 'Cabot+Guns+1911+pistol', 'Chamelot-Delvigne+Model+1873', 'Chauchat+light+machine+gun', 'CheyTac+Intervention+rifle', 'Chiappa+Rhino+40DS+revolver', 'Cobra+CA380', 'Cobra+FS380', 'Colt+1851+Navy+revolver', 'Colt+1873+Single+Action+Army', 'Colt+AR15+M4A1', 'Colt+Commando+rifle', 'Colt+Dragon+1848+Pocket+Pistol', 'Colt+LE6920', 'Colt+M1911', 'Colt+Mark+IV+Series+70', 'Colt+Model+1903+Pocket+Hammerless+pistol', 'Crosman+C-TT+pistol', 'DPMS+Oracle', 'Daewoo+K1', 'Degtyaryov+DPM+machine+gun', 'Derringer+Davis+Industries+22', 'Dornaus+Dixon+Bren+Ten', 'FMK+3', 'FMK+9C1', 'FN+FAL+rifle', 'FN+Five-seven+USG+pistol', 'FN+Model+1910+pistol', 'FN+SCAR+16S+rifle', 'FN+Tactical+Police+shotgun', 'FN+ps90+standard', 'FX+Gladiator+MKII', 'Famas+G2+rifle', 'Famas+Modele+F1', 'Franchi+SAS-12+shotgun', 'Freedom+Arms+model+83+revolver', 'Frommer+Modele+Stop+Calibre7,65mm', 'GSh-18+pistol', 'Galil+assault+rifle', 'Gasser+M1870', 'Glock+17+pistol', 'Heckler+Koch+G36+rifle', 'Heckler+Koch+G3A3+rifle', 'Heckler+Koch+HK33+rifle', 'Heckler+Koch+HK45+Compact+Tactical+pistol', 'Heckler+Koch+MG4+machine+gun', 'Heckler+Koch+MP5K-PDW+submachine+gun', 'Heckler+Koch+MP7A1+rifle', 'Heckler+Koch+Mark+23+pistol', 'Heckler+Koch+PSG1+rifle', 'Heckler+Koch+USP+compact', 'Ithaca+37', 'Ithaca+37+stakeout+shotgun', 'Jennings+Jimenez+Arms+Modele+J22', 'KS-23M+shotgun', 'Karabiner+98+kurz+rifle', 'Kel+Tec+KSG', 'Kel+Tec+SUB+2000', 'L42A1+sniper+rifle', 'Lanchester+submachine+gun', 'Lee-Enfield+No.4+Mk+I', 'Lewis+Gun', 'Llama+III+A', 'Lorcin+380', 'Luger+P08+pistol', 'M1+Carbine', 'M110+Semi-Automatic+Sniper+System', 'M14+SMUD+rifle', 'M16A4+rifle', 'M1911A1+pistol', 'M1928A1+Thompson+submachine+gun', 'M1941+Johnson+rifle', 'M24+Sniper+Weapon+System', 'M249S+semiautomatic+rifle', 'M27+Infantry+Automatic+Rifle', 'M39+Enhanced+Marksman+Rifle', 'M3A1+Grease+Gun', 'M40A5+Sniper+Rifle', 'M60E3+machine+gun', 'M8+Flare+Pistol', 'MAB+Model+D+pistol', 'MAC+Mle+1950', 'MAC-10', 'MAS-36+CR39', 'MAS-38', 'MAT-49+submachine+gun', 'MATEBA+AutoRevolver+6-Home+Protection', 'MEU(SOC)+pistol', 'MG+42+machine+gun', 'MG34', 'MP+40+submachine+gun', 'MP+41+Schmeisse+submachine+gun', 'MP-443+Grach+pistol', 'Makarov+PMM+pistol', 'Manurhin+MR-73', 'Margolin+MCM+Pistol', 'Mark+XIX+Desert+Eagle+pistol', 'Marlin+1881', 'Marlin+Model+60+rifle', 'Marlin+No32+Standard+1875', 'Marlin+XX+Standard+1873', 'Mauser+C96+Red+9', 'Mauser+Modele+K98', 'McMillan+Tac-50+A1+rifle', 'Mk+12+Special+Purpose+Rifle', 'Mossberg+453T', 'Mossberg+model+505+shotgun', 'Nagant+M1895+revolver', 'Norinco+Modele+NP22+9x19mm', 'OTs-38+Stechkin+silent+revolver', 'PKP+Pecheneg+machine+gun', 'PP-91+KEDR+submachine+gun', 'PPS-43+submachine+gun', 'PPSh-41', 'PSM', 'PSS+silent+pistol', 'Philadelphia+Deringer+pistol', 'Pistolet+Webley+Scott+9mm', 'Pistolet+wz.+35+Vis+pistol', 'QBZ-95+Assault+Rifle', 'RPD+machine+gun', 'RPK-74', 'RUGER+MK+II+pistol', 'RUGER+SINGLE+SIX+CONVERTIBLE+revolver', 'Reck+Modele+R15', 'Remington+MSR', 'Remington+Model+10+shotgun', 'Remington+Model+1100+shotgun', 'Remington+Model+870+Wingmaster', 'Rhoner+Modele+SM+110', 'Ruger+AR+556', 'Ruger+Lightweight+Compact+Pistol', 'Ruger+No.1+rifle', 'Ruger+Red+Label', 'Ruger+new+model+super+Blackhawk+revolver', 'Ruger+speed+Six+revolver', 'Ruger+standard+pistol', 'SG+552+Commando+rifle', 'SR-2+Veresk+submachine+gun', 'SR-25+sniper+rifle', 'SV-98+sniper+rifle', 'SVD+Dragunov', 'SVT-40+rifle', 'Saiga-20+shotgun', 'Savage+1905+1907+1917', 'Sears+Roebuck+Firearms+Model+66', 'Serdyukov+SPS', 'Sig+Sauer+P226+SCT+pistol', 'Sig+Sauer+P239+pistol', 'Sig+Sauer+P938+pistol', 'Sig+Sauer+Pro+sp+2009+pistol', 'Sig+Sauer+SIG716+Patrol', 'Sig+Sauer+sp+2022+pistol', 'Smith+Wesson+M&P15', 'Smith+Wesson+M1917+revolver', 'Smith+Wesson+MP+Shield', 'Smith+Wesson+Model+10+and+19+Revolver', 'Smith+Wesson+Model+39+pistol', 'Smith+Wesson+Model+41+pistol', 'Smith+Wesson+Model+5904+pistol', 'Smith+Wesson+Model+60+revolver', 'Smith+Wesson+Model+629+revolver', 'Smith+Wesson+Model+67+revolver', 'Smith+Wesson+Modele+99+9x19mm', 'Smith+wesson+New+Model+No.3+revolver', 'Springfield+M1903A4+rifle', 'Springfield+XD+S', 'StG+44+rifle', 'Stag+Arms+Model+8TL', 'Sterling+PPL', 'Steyr+AUG+rifle', 'Steyr+M1912+pistol', 'Steyr+Mannlicher+Modele+M95', 'Suomi+modele+KP31+9x19mm', 'Tanfoglio+GT28', 'Tanfoglio+TA90', 'Taurus+DT+.357+Magnum+Revolver', 'Tavor+TAR-21+assault+rifle', 'Thompson+Center+Arms+Encore+Muzzleloading+Rifle', 'Thompson+Center+Contender+Pistol', 'Tokarev+TT-33', 'Uzi+pistol', 'VSS+Vintorez', 'Valtro+PM-5+Shotgun', 'Vektor+SS77', 'Walther+MPK+submachine+gun', 'Walther+P38', 'Walther+PP', 'Weatherby+Mark+V+rifle', 'Winchester+Model+1200+Police', 'Winchester+Model+1894', 'Winchester+Model+1903', 'Winchester+Model+1912', 'Winchester+Model+70', 'XM25+CDTE', 'XM29+OICW+rifle', 'XM8', 'Zastava+M90', 'sako+TRG-42+sniper+rifle', 'sten+Mark+3+submachine+gun']
weapons_list = ['001','002','003','004','005','006','007','008','009','010']

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


if __name__ == "__main__":

    global module_id_mapper, features, grads
    module_id_mapper, features, grads = {}, {}, {}

    # Set parameters
    data_size = 384
    fpn_size = 1536
    num_classes = 10
    num_selects = {'layer1': 256, 'layer2': 128, 'layer3': 64, 'layer4': 32}

    # Build model : the file best.pt can be downloaded at the following url: https://cloud.irit.fr/s/j3HzVaA6DxUXHbZ
    model = build_model(pretrainewd_path = "best.pt",
                        img_size = data_size,
                        fpn_size = fpn_size,
                        num_classes = num_classes,
                        num_selects = num_selects)

    # Load image : : this part should be changed to be able to get the image from the client side
    img_url = 'imgs_test/Red_Winged_Blackbird_0010_6386.jpg' 
    #img_url = 'imgs_test/Groove_Billed_Ani_0009_1522.jpg'
    #img_url = 'imgs_test/aks74u_013.jpeg'
    
    img_loader = ImgLoader(img_size = data_size)
    img, ori_img = img_loader.load(img_url)

    # Predict the top 5 model names and their probabilities
    img = img.unsqueeze(0)
    out = model(img)
    results = cal_backward(out, sum_type="softmax")

    # The following lines should be replaced with sending back to the client side the information put inside print(...)
    print(weapons_list[results[0]["class"]] + " predicted with " + str(round(100*results[0]["accuracy"],2)) + "%")
    print(weapons_list[results[1]["class"]] + " predicted with " + str(round(100*results[1]["accuracy"],2)) + "%")
    print(weapons_list[results[2]["class"]] + " predicted with " + str(round(100*results[2]["accuracy"],2)) + "%")
    print(weapons_list[results[3]["class"]] + " predicted with " + str(round(100*results[3]["accuracy"],2)) + "%")
    print(weapons_list[results[4]["class"]] + " predicted with " + str(round(100*results[4]["accuracy"],2)) + "%")
