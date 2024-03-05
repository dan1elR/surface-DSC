import os
import numpy as np
from surface_distance.metrics import compute_surface_dice_at_tolerance as compSDSC
from surface_distance.metrics import compute_average_surface_distance as compAvgSDist
from surface_distance.metrics import compute_surface_distances as compSurfDist
import nibabel as nib

predNifti = '/home/daniel/Documents/TUMRc1-pred-peryton-PTV-label.nii.gz'
gtNifti = '/home/daniel/Documents/TUMRc1-gt-peryton-PTV-label.nii.gz'
gt = nib.load(gtNifti)
pred = nib.load(predNifti)
predData = pred.get_fdata().astype(bool)
gtData = gt.get_fdata().astype(bool)
gtSpacing = abs(gt.affine[:3,:3].diagonal())
predSpacing = abs(pred.affine[:3,:3].diagonal())
print('Spacing gt = pred: ', np.all(gtSpacing == predSpacing))

surfDist = compSurfDist(gtData, predData, gtSpacing)
tau = 1.5
sDSC = compSDSC(surfDist,tau)
print(os.path.basename(predNifti), 'PTV')
print('Surface Dice: ', sDSC)
print('tau: ', tau,'mm')