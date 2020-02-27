from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import argparse
# import cv2

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--image")
parser.add_argument("--video")
parser.add_argument("--mask_category")
args = parser.parse_args()

if args.image is None and args.video is None:
    print("ValueError. Set '--image' or '--video' flag")
    exit()

config_file = args.config
checkpoint_file = args.checkpoint

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

if args.image:
    # test a single image and show the results
    img = args.image # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # # visualize the results in a new window
    # show_result(img, result, model.CLASSES)
    # or save the visualization results to image files
    show_result(img, result, model.CLASSES, show=False, out_file='result.jpg', mask_category=args.mask_category)
if args.video:
    # test a video and show the results
    video = mmcv.VideoReader(args.video)
    for frame in video:
        result = inference_detector(model, frame)
        result_frame = show_result(frame, result, model.CLASSES, wait_time=1)
