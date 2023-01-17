import os
import sys
import torch
import platform
import argparse
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models import FCDenseNets
from models.common import DetectMultiBackend

from utils import resize, transform
from utils.plots import save_one_box
from utils.general import (Profile, check_img_size, check_imshow, cv2, increment_path, non_max_suppression, scale_boxes,
                           xyxy2xywh, xywh2xyxy, clip_boxes, colorstr)
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages, LoadStreams


def run(
        weights=ROOT / 'weights/yolov5.pt',  # model path or triton URL
        weights_seg=ROOT / 'weights/FCDenseNet56',  # model.pth path(s)
        source=ROOT / 'data/images/container.jpg',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/keyhole.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        imgsz_seg=256,  # inference size
        thresh=0.85,  # confidence threshold
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        nosave=False,  # do not save images/videos
        project=ROOT / 'runs/test',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        vid_stride=1  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave
    webcam = source.isnumeric()

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model_seg_name = Path(weights_seg).stem
    print(f'\nLoading {model_seg_name}...')
    if model_seg_name in FCDenseNets.keys():
        model_seg = FCDenseNets[model_seg_name].eval().to(model.device)
    else:
        raise SystemExit('Unsupported type of FC-DenseNet model')

    if os.path.exists(weights_seg):
        model_seg.load_state_dict(torch.load(weights_seg))
        print('Successfully loaded weights\n')
    else:
        raise SystemExit('Failed to load weights')

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    dt_seg = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, _ in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=False, max_det=max_det)

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0 = path[i], im0s[i].copy()
            else:
                p, im0 = path, im0s.copy()

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            imc = im0.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if conf < thresh:
                        continue
                    one_box = np.ascontiguousarray(save_one_box(xyxy, imc, BGR=True, save=False))

                    xyxy = torch.tensor(xyxy).view(-1, 4)
                    b = xyxy2xywh(xyxy)  # boxes
                    b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
                    xyxy = xywh2xyxy(b).long()
                    clip_boxes(xyxy, imc.shape)

                    with dt_seg[0]:
                        im0_seg = resize(one_box, imgsz_seg)
                        im_seg = torch.unsqueeze(transform(im0_seg), dim=0).to(model.device)
                    with dt_seg[1]:
                        pred_seg = model_seg(im_seg)
                    with dt_seg[2]:
                        pred_seg = pred_seg.cpu().detach().numpy()
                        pred_seg = pred_seg.reshape(pred_seg.shape[-2:]) * 255
                        pred_seg = resize(pred_seg.astype('uint8'), max(one_box.shape[:2]))

                        _, binary = cv2.threshold(pred_seg, 0, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        area = []
                        for j in range(len(contours)):
                            area.append(cv2.contourArea(contours[j]))
                            contours[j][:, :, 0] += int(xyxy[0, 0])
                            contours[j][:, :, 1] += int(xyxy[0, 1])
                        max_idx = np.argmax(area)
                        m = cv2.moments(contours[max_idx])
                        x = int(m['m10'] / m['m00'])
                        y = int(m['m01'] / m['m00'])
                        cv2.circle(im0, (x, y), 3, (0, 255, 0), line_thickness)
                        cv2.drawContours(im0, contours, max_idx, (0, 255, 0), line_thickness)
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    print(f'YOLOv5: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    t_seg = tuple(x.t / seen * 1E3 for x in dt_seg)  # speeds per image
    print(
        f'FC-DenseNet: %.1fms pre-process, %.1fms inference, %.1fms localization per image at shape {(1, 3, imgsz_seg, imgsz_seg)}' % t_seg)
    print(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s.pt', help='model path')
    parser.add_argument('--weights-seg', type=str, default=ROOT / 'weights/FCDenseNet56.pth', help='model path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images/container.jpg', help='file/dir/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/hole.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--imgsz-seg', '--img-seg', '--img-size-seg', type=int, default=256, help='inference size')
    parser.add_argument('--thresh', type=float, default=0.85, help='confidence threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', type=bool, help='show results', default=False)
    parser.add_argument('--nosave', type=bool, help='do not save images/videos', default=False)
    parser.add_argument('--project', default=ROOT / 'runs/test', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', type=bool, help='existing project/name ok, do not increment', default=False)
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', type=bool, help='use FP16 half-precision inference', default=False)
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
