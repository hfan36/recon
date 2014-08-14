#Alter numbers, but not the variable names; respect value types.  The # symbol is a delimeter for comments.

#############################
#### Geometry Parameters ####
#############################
R	= 1105.7			# millimeters, x-ray source to detector distance (float)
Rf 	= 907.45			# millimeters, x-ray source to rotation axis (float)
Ry  = 1105.7;  		    # millimeters, R + dy
Dx	= 0					# 1.4698 millimeters, Detector offset in x direction (float)
Dz	= 0					# 0.01228 millimeters, Detector offset in z direction (float)
xangle	= 0       		# -0.2004    radians, theta, Detector misalignment angle about x-axis (float)
yangle	= 0				# 0.0039     radians, eta, Detector misalignment angle about y-axis (float)
zangle	= 0				# -0.004     radians, phi, Detector misalignment angle about z-axis (float)
M 	= 13.10239     		# magnification of the lens (float)


#############################
#### Detector Parameters ####
#############################

TAxPixels	= 512	 	# 2160 Number of Transaxial pixels (unsigned int)
AxPixels	= 1024	 	# 1024, 2560 Number of Axial pixels (unsigned int)
TAxPixDim	= 0.0065 	# 0.0065 , 0.013 millimeters, Dimension of pixel in Transaxial direction (float)
AxPixDim	= 0.0065  	# 0.0065 , 0.013 millimeters, Dimension of pixel in Axial direction (float)

#############################
## Voxel Volume Parameters ##
#############################

Xdim		= 128		# Number of voxels in x direction (unsigned int)
Ydim		= 128		# Number of voxels in y direction (unsigned int)
Zdim		= 4			# Number of voxels in z direction (unsigned int)
XVoxSize	= 0.5 		# 1.5789  millimeters, size of voxel element in x direction (float)
YVoxSize	= 0.5		# 1.5789  millimeters, size of voxel element in y direction (float)
ZVoxSize	= 0.5		# 1.5789  millimeters, size of voxel element in z direction (float)

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

N_iteration = 1 	# Number of MLEM iterations (unsigned int);
NumProj		= 180  	# Number of projections in scan (unsigned int)
DeltaAng	= 2		# degrees, Angular spacing between projections (float)


#############################
######## File Paths #########
#############################

ProjFileFolder 		= H:\Visual Studio 2010\CT_Recon\CT_Recon\ball		# Folder path
ProjFileNameRoot 	= fp_ball64											# Projection File Root name
ProjFileSuffix 		= .bin												# Projection File suffix
ReconFileFolder 	= 													# Folder path for the reconstructed file
ReconFileNameRoot 	= recon_fp_ball64									# Reconstructed file root name
ReconFileSuffix 	= .bin												# Reconstructed file suffix

ObjFile	= lego_2x3_512.bin		# File path for object data (string)
ImgFile	= imageData.dat			# File path for image data (string)



