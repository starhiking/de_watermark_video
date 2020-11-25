import cv2
from estimate_watermark import estimate_watermark
from reconstruct_watermark import *

# gx, gy, gxlist, gylist = estimate_watermark("./resouces/filled")
gx, gy, gxlist, gylist, rect_start, rect_end = estimate_watermark(os.path.join('resources','raw'))
# print(gx,gy,gxlist,gylist)
# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy, 0.5)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
img = cv2.imread(os.path.join('resources','filled','1.jpg'))
img = img[rect_start[1]:rect_end[1],rect_start[0]:rect_end[0],:]
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)
print(start)
print(end)

# plt.imshow(im)
# plt.show()
# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(os.listdir(os.path.join('resources','filled')))

J, img_paths = get_cropped_images(
    os.path.join('resources','filled'), num_images, start, end, cropped_gx.shape, rect_start, rect_end)
# get a random subset of J

# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i] * alpha[:, :, i]

Wm = Wm + alpha * est_Ik

W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]

Jt = J[:25]
# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)

for i in range(11):
    img = cv2.imread((os.path.join('resources','filled','{}.jpg'.format(i+1))))
    img_rect = img[rect_start[1]:rect_end[1],rect_start[0]:rect_end[0],:]
    naive_Jt = img_rect[start[0]:(start[0]+end[0]), start[1]:(start[1]+end[1]), :]

    assert naive_Jt.shape == Ik[i].shape , "the size are not equal."

    img_rect[start[0]:(start[0]+end[0]), start[1]:(start[1]+end[1]), :] = Ik[i].copy()
    

    cv2.imwrite(os.path.join('resources','watermark','{}.jpg'.format(i+1)), img)
