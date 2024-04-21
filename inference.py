from pathlib import Path
from loguru import logger
import numpy as np
import torch
import time
import sys
import cv2
import os

from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp

# from boxmotxx import StrongSORT
# from boxmotxx import OCSORT
# from boxmotxx import BYTETracker
# from boxmotxx import BoTSORT
# from boxmotxx import DeepOCSORT
# from boxmotxx import HybridSORT

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

def create_predictor(args):
    exp = get_exp(args.exp_file, args.name)

    if args.trt:
        args.device = "gpu"

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = "model_trt.pth"
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    if args.cls_names:
        cls_names = args.cls_names
    else:
        cls_names = COCO_CLASSES

    return Predictor(
        model, exp, cls_names, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )

class Args:
    def __init__(self):
        self.cls_names = None
        self.name = "yolox-x"
        self.ckpt = "weights/yolox_x.pth"
        self.exp_file = None
        self.device = "gpu"
        # self.device = "mps"
        # self.device = "cpu"
        self.conf = 0.01
        self.nms = 0.3
        self.tsize = 640
        self.fp16 = False
        self.legacy = False
        self.fuse = False
        self.trt = False







### avaible ReID models
# reid_model = 'lmbn_n_cuhk03_d.pt'        # lightweight
# reid_model = 'osnet_x0_25_market1501.pt'
# reid_model = 'mobilenetv2_x1_4_msmt17.engine'
# reid_model = 'resnet50_msmt17.onnx'
reid_model = 'osnet_x1_0_msmt17.pt'
# reid_model = 'clip_market1501.pt'        # heavy
# reid_model = 'clip_vehicleid.pt'
reid_device = 'gpu'
reid_fp16 = True

# tracker = StrongSORT( # why this has no, plot_results
#       model_weights=Path(reid_model),
#       device=reid_device,
#       fp16=reid_fp16,
# )
# tracker = OCSORT(
# )
# tracker = BYTETracker(
# )
# tracker = BoTSORT(
#       model_weights=Path(reid_model),
#       device=reid_device,
#       fp16=reid_fp16,
# )
# tracker = DeepOCSORT(
#       model_weights=Path(reid_model),
#       device=reid_device,
#       fp16=reid_fp16,
# )
# tracker = HybridSORT(
#       Path(reid_model),
#       device=reid_device,
#       half=reid_fp16,
#     det_thresh=0.12442660055370669,
# )

predictor = None

def track_frame(image):
        outputs, img_info = predictor.inference(image)
        output = outputs[0]
        if output is None:
                return [], [], []
        output = output.cpu()

        bboxes = output[:, 0:4]
        ratio = img_info["ratio"]
        bboxes /= ratio
        bboxes = bboxes.tolist()

        cls = output[:, 6]
        cls = cls.tolist()
        cls = [int(x) for x in cls]

        scores = output[:, 4] * output[:, 5]
        scores = scores.tolist()
        return bboxes, cls, scores

def main():
    global predictor
    args = Args()
    predictor = create_predictor(args)

    vidcap = cv2.VideoCapture(sys.argv[1])

    success = True
    while success:
        success, frame = vidcap.read()
        if success:

            use_tracker = False

            if use_tracker:
                result_image = frame.copy()
                boxes, label_ids, confidences = track_frame(result_image)
                dets = []
                for i in range(len(boxes)):
                    new = boxes[i] + [confidences[i]] + [label_ids[i]]
                    dets.append(new)
                # if dets:
                #     tracker.update(np.array(dets), result_image)
                #     tracker.plot_results(result_image, show_trajectories=True)
            else:
                outputs, img_info = predictor.inference(frame)
                result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

            cv2.imshow('yolo', result_image)

        key = cv2.waitKey(30) & 0xFF
        if key == 27: # esc to quit
            break

if __name__ == "__main__":
    main()
