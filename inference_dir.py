from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import argparse
import os
import re
# import cv2

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--image_dir")
# parser.add_argument("--video")
parser.add_argument("--mask_category")
args = parser.parse_args()

if args.image_dir is None: # and args.video is None:
    print("ValueError. Set '--image_dir'") #　or '--video' flag")
    exit()

config_file = args.config
checkpoint_file = args.checkpoint

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


####
out_dir = "./result/"
os.makedirs(out_dir, exist_ok=True)
pattern = ".*\.(jpg)"
names = [f for f in os.listdir(args.image_dir) if re.search(pattern, f, re.IGNORECASE)] # 大小文字無視

for name in names:
    # test a single image and show the results
    img =  args.image_dir + "/" + name # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # # visualize the results in a new window
    # show_result(img, result, model.CLASSES)
    # or save the visualization results to image files
    os.makedirs('./masks', exist_ok=True)
    show_result(img, result, model.CLASSES, show=False, out_file=out_dir+'result_'+name, mask_category=args.mask_category)
    os.rename("./masks", out_dir+"masks_"+name)
# if args.video:
#     # test a video and show the results
#     video = mmcv.VideoReader(args.video)
#     for frame in video:
#         result = inference_detector(model, frame)
#         result_frame = show_result(frame, result, model.CLASSES, wait_time=1)
