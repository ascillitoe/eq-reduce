import pickle
import numpy as np
import pyvista as pv
import os
import shutil

pwd = os.getcwd()
orig_dir = 'ORIG_DATA'
new_dir  = 'SUBSAMPLED_DATA'

###################################################################
# Load data (and subsample)
###################################################################
orig_dir = os.path.join(pwd,orig_dir)
new_dir  = os.path.join(pwd,new_dir)

# Load baseline mesh (and subsample)
basegrid = pv.read(os.path.join(orig_dir,'basegrid.vtk'))
xskip = 3
yskip = 2
nx,ny,_ = basegrid.dimensions
npts = int(nx*ny)
idx = np.arange(npts).reshape(nx,ny,order='F')[::yskip,::xskip].T
points = basegrid.points_matrix[:,:,0,:2].transpose([1,0,2])
pts = idx.flatten()
x = points[:,0,0][::xskip]
y = points[0,:,1][::yskip]
assert len(x)*len(y)==len(pts)

# Load and index with `pts` array to subsample
# Load poly coeffs etc
coeffs = np.load(os.path.join(orig_dir,'coeffs.npy'))[pts] 
lowers = np.load(os.path.join(orig_dir,'lowers.npy'))[pts]
uppers = np.load(os.path.join(orig_dir,'uppers.npy'))[pts]
W      = np.load(os.path.join(orig_dir,'W.npy'))[pts]

# Load training data to plot on summary plots
Y = np.load(os.path.join(orig_dir,'Y.npy'))[pts,:,:]

###################################################################
# Save subsampled data
###################################################################
np.save(os.path.join(new_dir,'coeffs.npy'),coeffs)
np.save(os.path.join(new_dir,'lowers.npy'),lowers)
np.save(os.path.join(new_dir,'uppers.npy'),uppers)
np.save(os.path.join(new_dir,'W.npy'),W)
np.save(os.path.join(new_dir,'Y.npy'),Y)
np.save(os.path.join(new_dir,'xpts.npy'),x)
np.save(os.path.join(new_dir,'ypts.npy'),y)

###################################################################
# Copy across other files that don't need subsampling
###################################################################
shutil.copyfile(os.path.join(orig_dir,'surface_base.vtk'),os.path.join(new_dir,'surface_base.vtk'))
shutil.copyfile(os.path.join(orig_dir,'X.npy'),os.path.join(new_dir,'X.npy'))
