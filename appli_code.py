import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from io import BytesIO
from PIL import Image

weapons_list = ['00001_Sig_Sauer_P228', '00002_Sig_Sauer_P938', '00003_Sig_Sauer_SP2022', '00004_Smith_and_Wesson_Model_19', '00005_Smith_and_Wesson_5904', '00006_Smith_and_Wesson_M&P_Shield', '00007_Smith_and_Wesson_M&P15', '00008_Smith_and_Wesson_Model_10_Revolver', '00009_Smith_and_Wesson_Modele_99_9x19mm', '00010_FMK_3', '00011_FMK_9C1', '00012_Colt_Mark_IV_Series_70', '00013_Colt_LE6920', '00014_Colt_M1911', '00015_Colt_AR15_M4A1', '00016_Colt_AR15_M16A1', '00017_Beretta_92FS', '00018_Beretta_98FS', '00019_Mossberg_453T', '00020_Mossberg_500_tactical', '00021_Mossberg_500', '00022_Mossberg_590A1', '00023_Thompson_Center_Arms_Encore_Muzzleloading_Rifle', '00024_Bushmaster_AR_15_Semiautomatic_Rifle', '00025_Ruger_AR_556', '00026_Ruger_Lightweight_Compact_Pistol', '00027_Ruger_Mark4', '00028_Kel_Tec_SUB_2000', '00029_Kel_Tec_KSG', '00030_Walther_modele_PPK', '00031_Astra_Modele_900', '00032_Astra_Modele_903', '00033_Frommer_Modele_Stop_Calibre7,65mm', '00034_Jennings_Jimenez_Arms_Modele_J_22', '00035_Manurhin_Mod_MR73', '00036_Mauser_Modele_C96', '00037_Mauser_Modele_K98', '00038_Mitrailleuse_MG42', '00039_Remington_870', '00040_Norinco_Modele_NP22_9x19mm', '00041_Pistolet_Webley&Scott_9mm', '00042_Reck_Modele_R15', '00043_Rhoner_Modele_SM_110', '00044_Llama_III_A', '00045_Steyr_Mannlicher_Modele_M95', '00046_STG44_7,92x33mm', '00047_Suomi_modele_KP31_9x19mm', '00048_Tanfoglio_GT28', '00049_Tanfoglio_TA90', '00050_Famas_Modele_F1', '00051_Famas_Modele_G2', '00052_AK47', '00053_AK74', '00054_AKS74U', '00055_AK101', '00056_SVD_Dragunov', '00057_ArmaLite_AR10', '00058_ArmaLite_AR15', '00059_PSM', '00060_Margolin_MCM_Pistol', '00061_Barrett_M95', '00062_Ithaca37', '00063_DPMS_Oracle', '00064_Daewoo_K1', '00065_Vektor_SS77', '00066_MP40', '00067_Glock_17', '00068_Glock_19', '00069_Glock_22', '00070_Glock_25', '00071_Glock_31', '00072_Glock_37', '00073_IWI_Tavor_21', '00074_Zastava_M90', '00075_Winchester_Model_1200', '00076_Winchester_Model_1887', '00077_Winchester_Model_1892', '00078_Winchester_Model_1894', '00079_Winchester_Model_1897', '00080_Winchester_Model_1903', '00081_Marlin_1881', '00082_Marlin_1893', '00083_Marlin_XXX_Standard_1872', '00084_Marlin_XX_Standard_1873', '00085_Marlin_No32_Standard_1875', '00086_Johnson_M1941', '00087_Lorcin_380', '00088_Cobra_FS380', '00089_Cobra_CA380', '00090_Springfield_XD_S', '00091_Bauer_Automatic_25', '00092_Savage_1905', '00093_Savage_1907', '00094_Savage_1917', '00095_Sterling_PPL', '00096_Derringer_Davis_Industries_22', '00097_Dornaus_and_Dixon', '00098_Benelli_M4_Super_90', '00099_Gasser_M1870', '00100_Sears_Roebuck_Firearms_Model_66']

# Loading the file containing the weights of the trained network
network_file_url = './records/FGVC-HERBS/ogs/backup/best.pt'  #'https://github.com/keurstudio/firearm_webapp/raw/main/reseau.h5'
network_file = requests.get(network_file_url, allow_redirects=True)

# Writing the file on the disk to load the model from the file
open('network_weights.h5', 'wb').write(network_file.content)
model = keras.models.load_model('network_weights.h5')

# Getting an image to predict : this part should be changed to be able to get the image from the client side
img_url = 'https://cloud10.todocoleccion.online/militaria-armas-fuego/tc/2023/01/08/22/385386989_tcimg_E91D83C4.jpg'
res = requests.get(img_url)
img = Image.open(BytesIO(res.content))

# Preprocessing the image (resizing and putting the image in a tensor)
img_size = 224

img = img.resize((img_size,img_size), Image.ANTIALIAS)
img = np.asarray(img)

tensor = np.zeros((1, img_size, img_size, 3))
tensor[0] = img

# Predict the top 5 model names and their probabilities
eval = model.predict(tensor, verbose=0)
ordre = -np.sort(-eval)[:,0:5]
args = np.flip(np.argsort(eval), 1)[:,0:5]

# The following lines should be replaced with sending back to the client side the information put inside print(...)
print(weapons_list[args[0][0]], ordre[0][0])
print(weapons_list[args[0][1]], ordre[0][1])
print(weapons_list[args[0][2]], ordre[0][2])
print(weapons_list[args[0][3]], ordre[0][3])
print(weapons_list[args[0][4]], ordre[0][4])
