import cv2
from estimate_watermark import estimate_watermark_video 
from reconstruct_watermark import *

video = "video\B2.mp4"
out_video = 'video\\output4.avi'
_GAP = 50 # frequence
img_size = 512

cap = cv2.VideoCapture(video)
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MPEG') , video_fps, video_size)

# gx, gy, gxlist, gylist = estimate_watermark("./resouces/filled")
gx, gy, gxlist, gylist, rect_start, rect_end, frames = estimate_watermark_video(video,GAP=_GAP,img_size=img_size)
# print(gx,gy,gxlist,gylist)
# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy, 0.4)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
im, start, end = watermark_detector(frames[0], cropped_gx, cropped_gy)
print(start)
print(end)

# plt.imshow(im)
# plt.show()
# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(frames)

J = get_cropped_images_video(frames, start, end, cropped_gx.shape)
# get a random subset of J

# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm, num_images)
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


index = 0
I_rect = None
result_frames = []
count_frame = 0

while True:
    result,frame = cap.read()
    if not result: break
    # if img_size > 0:
    #     frame = cv2.resize(frame,(img_size,img_size))

    I_rect = Ik[index].copy()

    if index % _GAP is 0:    
        index+=1
        
    _start_y = rect_start[0] + start[0]
    _start_x = rect_start[1] + start[1]
    _end_y = _start_y + end[1]
    _end_x = _start_x + end[0]

    if img_size > 0 :
        size_ratio = np.array(frame.shape[:2]) / img_size
        _start_y = (_start_y * size_ratio[1]).astype(np.int)
        _start_x = (_start_x * size_ratio[0]).astype(np.int)
        _end_y = (_end_y * size_ratio[1]).astype(np.int)
        _end_x = (_end_x * size_ratio[0]).astype(np.int)
        
        I_rect = cv2.resize(I_rect,(_end_y-_start_y,_end_x-_start_x))

        
    # naive_Jt = frame[_start_x:_end_x,_start_y:_end_y,:]
    
    # assert naive_Jt.shape == I_rect.shape
    frame[_start_x:_end_x,_start_y:_end_y,:] = (PlotImage(I_rect)*255).astype(np.uint8)

    writer.write(frame)
    # cv2.imwrite(os.path.join('resources','watermark','{}.jpg'.format(count_frame+1)), frame)
    count_frame += 1

cv2.destroyAllWindows()
cap.release()
writer.release()

print("Total frames : {}".format(count_frame))