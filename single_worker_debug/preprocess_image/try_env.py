import gym
import random
import cv2
import numpy as np

# env = gym.make('PongDeterministic-v4')
# obs = env.reset()
#
# # print(obs[100, 100, :])
#
# # cv2.imshow('pic', obs[:,:,::-1])
# # cv2.waitKey(0)
# step = 0
# while True:
#     action = random.sample([0, 2, 5], 1)
#     next_obs, _, _, _ = env.step(action)
#     rgb = env.render('rgb_array')
#     print(next_obs[100, 100, :])
#     print(rgb[100, 100, :])
#     step += 1
#     if step == 18:
#         cv2.imshow('rgb', rgb[:,:,::-1])
#         cv2.imwrite('raw.png', rgb[:, :, ::-1])
#         cv2.waitKey(3)

pic = cv2.imread('raw.png')
# crop image
pic = pic[34:34 + 160, :160]
pic_1 = cv2.resize(pic, (80, 80))
pic_2 = cv2.resize(pic_1, (42, 42))
gray_pic = pic_2.mean(2, keepdims=True)
gray_pic *= (1.0 / 255.0)

save_pic = gray_pic.reshape(42, 42)
print(gray_pic)
np.savetxt('gray_pic_data.csv', save_pic, delimiter=',')
cv2.imshow('gray img', gray_pic)
cv2.imwrite('gray_img_scale.png', gray_pic)
cv2.waitKey(0)
