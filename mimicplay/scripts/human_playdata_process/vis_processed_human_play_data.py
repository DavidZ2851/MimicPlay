import h5py
import numpy as np
import cv2
import pdb

view_id = 1 # change to 2 if drawing on second view


with h5py.File('demo_hand_loc_1_new_720_720.hdf5', 'r') as f:
    images = np.array(f['data/demo_0/obs/front_image_{}'.format(view_id)])
    actions = np.array(f['data/demo_0/actions'])

# Reshape the actions to [145, 10, 4]
actions = actions.reshape((-1, 10, 4))

H, W, _ = images[0].shape

if view_id == 1:
    actions = actions[:, :, :2]
else:
    actions = actions[:, :, 2:]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (W, H))
import pdb
pdb.set_trace()
img_size = 720
for i in range(images.shape[0]):
    img = images[i].copy()
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (img_size, img_size))

    action = actions[i]
    pdb.set_trace()
    action_unscaled = action * np.array([img.shape[1], img.shape[0]])

    for pt in action_unscaled:
        img = cv2.circle(img, (int(pt[1]), int(pt[0])), radius=5, color=(0, 255, 0), thickness=-1)

    video.write(img)

video.release()

print("The video has been successfully saved as output.mp4")