#Alter numbers, but not the variable names; respect value types.  The # symbol is a delimeter for comments.

#############################
#### Geometry Parameters ####
#############################
R	= 1105.7			# millimeters, x-ray source to detector distance (float)
Rf 	= 907.45			# millimeters, x-ray source to rotation axis (float)
Ry  = 1105.7;  		    # millimeters, R + dy
Dx	= 1.4698			# 1.4698 millimeters, Detector offset in x direction (float)
Dz	= 0.01228			# 0.01228 millimeters, Detector offset in z direction (float)
xangle	= -0.2004  		# -0.2004    radians, theta, Detector misalignment angle about x-axis (float)
yangle	= 0.0039		# 0.0039     radians, eta, Detector misalignment angle about y-axis (float)
zangle	= -0.004		# -0.004     radians, phi, Detector misalignment angle about z-axis (float)
M 	= 13.10239     		# magnification of the lens (float)


#############################
#### Detector Parameters ####
#############################

TAxPixels	= 1024	 	# 2160 Number of Transaxial pixels (unsigned int)
AxPixels	= 1024	 	# 1024, 2560 Number of Axial pixels (unsigned int)
TAxPixDim	= 0.0065 	# 0.0065 , 0.013 millimeters, Dimension of pixel in Transaxial direction (float)
AxPixDim	= 0.0065  	# 0.0065 , 0.013 millimeters, Dimension of pixel in Axial direction (float)

#############################
## Voxel Volume Parameters ##
#############################

Xdim		= 512		# Number of voxels in x direction (unsigned int)
Ydim		= 512		# Number of voxels in y direction (unsigned int)
Zdim		= 10		# Number of voxels in z direction (unsigned int)
XVoxSize	= 0.25			# 1.5789  millimeters, size of voxel element in x direction (float)
YVoxSize	= 0.25			# 1.5789  millimeters, size of voxel element in y direction (float)
ZVoxSize	= 0.25			# 1.5789  millimeters, size of voxel element in z direction (float)

##############################################
##### Object Parameters (not really used)#####
##############################################

ObjX	  = 0		# millimeters, x shift of object from origin (float)
ObjY	  = 32	    # millimeters, y shift of object from origin (float)
ObjZ	  = 32	    # millimeters, z shift of object from origin (float)
ObjXangle = 0.0		# degrees, angular orientation of object about x-axis (float)
ObjYangle = 0.0		# degrees, angular orientation of object about y-axis (float)
ObjZangle = 0.0		# degrees, angular orientation of object about z-axis (float)


#############################
##### Scan Parameters #######
#############################

N_iteration = 20 	# Number of MLEM iterations (unsigned int);
NumProj		= 180  	# Number of projections in scan (unsigned int)
DeltaAng	= 2		# degrees, Angular spacing between projections (float)


#############################
######## File Paths #########
#############################

ProjFileFolder 		= H:\CT data\051314\cleaned_data					# Folder path
ProjFileNameRoot 	= C1024_2_clean_phantom_60kV_400uA_10sec			# Projection File Root name
ProjFileSuffix 		= .ct												# Projection File suffix
ReconFileFolder 	= 													# Folder path for the reconstructed file
ReconFileNameRoot 	= recon_phantom512_s10										# Reconstructed file root name
ReconFileSuffix 	= .bin												# Reconstructed file suffix

#####################################################
##### File Paths for simulation purposes only #######
#####################################################

ObjFileFolder			= H:\Visual Studio 2010\CT_Recon\CT_Recon			
ObjFileName				= phantom_voxelvolume.bin						

BpFileNameRoot 			 = bp_ball64
BpFileSuffix 			 = .bin


## The Forward and Backward simulation will also use ProjFileFolder, ProjFileNameRoot and ProjFileSuffix 

## For forward projection, the object file is loaded using the names indicated in the simulation section. 
## Once the projections are calculated, they will be saved as: 
## ProjFileFolder + ProjFileNameRoot + projection loop increment + ProjFileSuffix 

## For backward projection, the each projection image file is loaded according to: 
## ProjFileFolder + ProjFileNameRoot + projection loop increment + ProjFileSuffix 
## The backward projection files are saved according to BpFileNameRoot and BpFileSuffix, 
## where if indicated, each bp file at each angle is save separately as: 
## ProjFileFolder + BpFileNameRoot + projection loop increment + BpFileSuffix. 
## The total bp for all angle is saved as:
## ProjFileFoler + BpFileNameRoot + BpFileSuffix. 

## !!!NOTE!!!: ReconFile* does not effect simulations







