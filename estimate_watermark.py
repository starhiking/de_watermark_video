import cv2
import cv2 as cv
import glob
import os
import math
import numpy as np
import scipy as sp
import scipy.fftpack
from matplotlib import pyplot as plt


KERNEL_SIZE = 3

def SetPoints(windowname, img):
    """
    输入图片，打开该图片进行标记点，返回的是标记的几个点的字符串
    """
    print('(提示：单击需要标记的坐标，Enter确定，Esc跳过，其它重试。)')
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(temp_img, (x, y), 5, (102, 217, 239), -1)
            points.append([x, y])
            cv.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv.namedWindow(windowname)
    cv.imshow(windowname, temp_img)
    cv.setMouseCallback(windowname, onMouse)
    key = cv.waitKey(0)
    if key == 13:  # Enter
        print('坐标为：', points)
        del temp_img
        cv.destroyAllWindows()
        return np.array(points).reshape(-1,2)
    elif key == 27:  # ESC
        print('跳过该张图片')
        del temp_img
        cv.destroyAllWindows()
        return None
    else:
        print('重试!')
        return SetPoints(windowname, img)

def estimate_watermark_video(video,GAP,img_size=256):
	"""
	estimate the watermark using, grad(W) = median(grad(J))
	"""
	if not os.path.exists(video):
		print("[estimate watermark] video {} not exist.".format(video))
		exit(1)
		return

	cap = cv2.VideoCapture(video)

	index = 0
	frames = []
	total_size = img_size * img_size * 3
	rect_start = None
	rect_end = None
	get_rect = False

	while True:
		result, frame = cap.read()
		if not result: break

		if index % GAP is 0:
			if frame.size > total_size:
				if img_size > 0 :
					frame = cv2.resize(frame,(img_size,img_size))
				if get_rect:
					frame = frame[rect_start[1]:rect_end[1],rect_start[0]:rect_end[0],:]
				frames.append(frame)
			
			if not get_rect:
				points = SetPoints("getPoint",frame)
				if points is None or points.shape[0]<2:
					continue

				rect_start = np.min(points,0)
				rect_end = np.max(points,0)
				get_rect = True

				for i in range(len(frames)):
					total_img = frames[i]
					frames[i] = total_img[rect_start[1]:rect_end[1],rect_start[0]:rect_end[0],:]
		index += 1


	for i in range(min(10,len(frames))):
		show_img = frames[i]
		cv2.imshow(str(i),show_img)

	cv2.waitKey(500)
	cv.destroyAllWindows()

	print("[estimate watermark] compute gradients.")
	grad_x = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), frames))
	grad_y = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), frames))

	print("[estimate watermark] compute median of all gradients.")
	Wm_x = np.median(np.array(grad_x), axis=0)
	Wm_y = np.median(np.array(grad_y), axis=0)

	return (Wm_x, Wm_y, grad_x, grad_y, rect_start, rect_end, frames)


def estimate_watermark(folder):
	"""
	estimate the watermark using, grad(W) = median(grad(J))
	"""
	if not os.path.exists(folder):
		print("[estimate watermark] folder {} not exist.".format(folder))
		exit(1)
		return

	ims = []
	rect_start = None
	rect_end = None
	get_rect = False

	for img in os.listdir(folder): # glob.glob(os.path.join('resources','raw','*.jpg')):
		img = os.path.join(folder,img)
		im = cv2.imread(img)
		if im is not None:
			if get_rect:
				im = im[rect_start[1]:rect_end[1],rect_start[0]:rect_end[0],:]
				if im is []:
					print(cv2.imread(img).shape)
			ims.append(im)
			assert im.shape == ims[0].shape
		else:
			print("[estimate watermark] image {} not exist.".format(img))
		
		if not get_rect :
			points = SetPoints("getPoint",im)
			if points is None or points.shape[0]<2:
				continue

			rect_start = np.min(points,0)
			rect_end = np.max(points,0)
			get_rect = True

			for i in range(len(ims)):
				total_img = ims[i]
				ims[i] = total_img[rect_start[1]:rect_end[1],rect_start[0]:rect_end[0],:]

	for i in range(min(10,len(ims))):
		img = ims[i]
		cv2.imshow(str(i),img)
	
	cv2.waitKey(500)
	cv.destroyAllWindows()


	print("[estimate watermark] compute gradients.")
	grad_x = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), ims))
	grad_y = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), ims))

	print("[estimate watermark] compute median of all gradients.")
	Wm_x = np.median(np.array(grad_x), axis=0)
	Wm_y = np.median(np.array(grad_y), axis=0)

	return (Wm_x, Wm_y, grad_x, grad_y, rect_start, rect_end)


def PlotImage(image):
	""" 
	PlotImage: Give a normalized images matrix which can be used with implot, etc.
	Maps to [0, 1]
	"""
	im = image.astype(float)
	return (im - np.min(im))/(np.max(im) - np.min(im))


def poisson_reconstruct2(gradx, grady, boundarysrc):
	# Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
	# Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

	# Laplacian
	gyy = grady[1:,:-1] - grady[:-1,:-1]
	gxx = gradx[:-1,1:] - gradx[:-1,:-1]
	f = np.zeros(boundarysrc.shape)
	f[:-1,1:] += gxx
	f[1:,:-1] += gyy

	# Boundary images
	boundary = boundarysrc.copy()
	boundary[1:-1,1:-1] = 0

	# Subtract boundary contribution
	f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
	f = f[1:-1,1:-1] - f_bp

	# Discrete Sine Transform
	tt = scipy.fftpack.dst(f, norm='ortho')
	fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

	# Eigenvalues
	(x,y) = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
	denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

	f = fsin/denom

	# Inverse Discrete Sine Transform
	tt = scipy.fftpack.idst(f, norm='ortho')
	img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

	# New center + old boundary
	result = boundary
	result[1:-1,1:-1] = img_tt

	return result


def poisson_reconstruct(gradx, grady, kernel_size=KERNEL_SIZE, num_iters=100, h=0.1, 
		boundary_image=None, boundary_zero=True):
	"""
	Iterative algorithm for Poisson reconstruction. 
	Given the gradx and grady values, find laplacian, and solve for images
	Also return the squared difference of every step.
	h = convergence rate
	"""
	fxx = cv2.Sobel(gradx, cv2.CV_64F, 1, 0, ksize=kernel_size)
	fyy = cv2.Sobel(grady, cv2.CV_64F, 0, 1, ksize=kernel_size)
	laplacian = fxx + fyy
	m,n,p = laplacian.shape

	if boundary_zero == True:
		est = np.zeros(laplacian.shape)
	else:
		assert(boundary_image is not None)
		assert(boundary_image.shape == laplacian.shape)
		est = boundary_image.copy()

	est[1:-1, 1:-1, :] = np.random.random((m-2, n-2, p))
	loss = []

	for i in range(num_iters):
		old_est = est.copy()
		est[1:-1, 1:-1, :] = 0.25*(est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] + est[1:-1, 2:, :] - h*h*laplacian[1:-1, 1:-1, :])
		error = np.sum(np.square(est-old_est))
		loss.append(error)

	return (est)


def image_threshold(image, threshold=0.5):
	'''
	Threshold the images to make all its elements greater than threshold*MAX = 1
	'''
	m, M = np.min(image), np.max(image)
	im = PlotImage(image)
	im[im >= threshold] = 1
	im[im < 1] = 0
	return im


def crop_watermark(gradx, grady, threshold=0.4, boundary_size=2):
	"""
	Crops the watermark by taking the edge map of magnitude of grad(W)
	Assumes the gradx and grady to be in 3 channels
	@param: threshold - gives the threshold param
	@param: boundary_size - boundary around cropped images
	"""
	W_mod = np.sqrt(np.square(gradx) + np.square(grady))
	W_mod = PlotImage(W_mod)
	W_gray = image_threshold(np.average(W_mod, axis=2), threshold=threshold)
	x, y = np.where(W_gray == 1)

	xm, xM = np.min(x) - boundary_size - 1, np.max(x) + boundary_size + 1
	ym, yM = np.min(y) - boundary_size - 1, np.max(y) + boundary_size + 1

	xm = max(0,xm)
	ym = max(0,ym)
	xM = min(xM,gradx.shape[0]-1)
	yM = min(yM,grady.shape[1]-1)


	return gradx[xm:xM, ym:yM, :] , grady[xm:xM, ym:yM, :]


def normalized(img):
	"""
	Return the images between -1 to 1 so that its easier to find out things like
	correlation between images, convolutionss, etc.
	Currently required for Chamfer distance for template matching.
	"""
	return (2*PlotImage(img)-1)


def watermark_detector(img, gx, gy, thresh_low=200, thresh_high=220, printval=False):
	"""
	Compute a verbose edge map using Canny edge detector, take its magnitude.
	Assuming cropped values of gradients are given.
	Returns images, start and end coordinates
	"""
	Wm = (np.average(np.sqrt(np.square(gx) + np.square(gy)), axis=2))

	img_edgemap = (cv2.Canny(img, thresh_low, thresh_high))
	chamfer_dist = cv2.filter2D(img_edgemap.astype(float), -1, Wm)

	rect = Wm.shape
	index = np.unravel_index(np.argmax(chamfer_dist), img.shape[:-1])
	if printval:
		print(index)

	x, y = int(index[0]-rect[0]/2), int(index[1]-rect[1]/2)  # 必须要为整数，所以做出调整
	im = img.copy()
	cv2.rectangle(im, (y, x), (y+rect[1], x+rect[0]), (255, 0, 0))
	return (im, (x, y), (rect[0], rect[1]))
