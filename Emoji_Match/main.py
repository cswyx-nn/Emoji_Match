import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
import emotion_classification as ec
from load_emoji import add_emoji
import sys


def main(text_label=True, rectangle=True, source='http://192.168.1.144:4747/video'):
    # Set the execution device and give priority to the use of GPU.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    weights = 'detect_weights/best.pt'                  # Set weights file path
    model = attempt_load(weights, map_location=device)  # load trained model
    stride = int(model.stride.max())                    # model stride

    # Set image source and check source type
    imgsz = 640
    source = source
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    imgsz = check_img_size(imgsz, s=stride)

    # Read image source
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Set half-precision floating-point number
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    half = device.type != 'cpu'
    if half:
        model.half()

    # Start using Model for face detection and expression classification
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Face detection
        pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

        for i, det in enumerate(pred):  # detections per image
            # Get bounding box form predication result
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            bboxes = det.cpu().numpy()

            # Classify the facial expressions in each bounding box
            for bbox in bboxes:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                bbox_img = im0[y1:y2, x1:x2]   # Extract the image in the bounding box
                bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2GRAY)
                # Perform emotion classification task
                emotion_label = ec.emotion_classification(bbox_img, image_size=48, channel=1)
                # Add emoji to image according to the result of emotion classification
                im0 = add_emoji(im0, emotion_label, [x1, y1, x2, y2])
                # Whether show emotion label
                if text_label:
                    cv2.putText(im0, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                # Whether show the face bounding box
                if rectangle:
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            if webcam:
                # Video Show
                out_win = 'Result_Window'
                cv2.imshow(out_win, im0)
                cv2.waitKey(1)
            else:
                if source.endswith('.mp4'):
                    out_win = 'Result_Window'
                    cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow(out_win, im0)
                    cv2.waitKey(2)
                else:
                    cv2.imwrite(path.split('.')[0] + '_result.jpg', im0)


if __name__ == '__main__':
    text_label = True if sys.argv[1] == '1' else False
    rectangle = True if sys.argv[2] == '1' else False
    source = sys.argv[3]
    main(text_label=text_label, rectangle=rectangle, source=source)
