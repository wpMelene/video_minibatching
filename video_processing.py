import numpy as np
from threading import Thread
import os
import cv2
import magic
import time

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class LoadStreamsBatch:

    def __init__(self, sources='streams.txt', img_size=416, batch_size=2, subdir_search=False):
        self.mode = 'images'
        self.img_size = img_size
        self.def_img_size = None

        videos = []
        if os.path.isdir(sources):
            if subdir_search:
                for subdir, dirs, files in os.walk(sources):
                    for file in files:
                        if 'video' in magic.from_file(subdir + os.sep + file, mime=True):
                            videos.append(subdir + os.sep + file)
            else:
                for elements in os.listdir(sources):
                    if not os.path.isdir(elements) and 'video' in magic.from_file(sources + os.sep + elements, mime=True):
                        videos.append(sources + os.sep + elements)
        else:
            with open(sources, 'r') as f:
                videos = [x.strip() for x in f.read().splitlines() if len(x.strip())]

        n = len(videos)
        curr_batch = 0
        self.data = [None] * batch_size
        self.cap = [None] * batch_size
        self.sources = videos
        self.n = n
        self.cur_pos = 0

        # Start the thread to read frames from the video stream
        for i, s in enumerate(videos):
            if curr_batch == batch_size:
                break
            print('%g/%g: %s... ' % (self.cur_pos+1, n, s), end='')
            self.cap[curr_batch] = cv2.VideoCapture(s)
            try:
                assert self.cap[curr_batch].isOpened()
            except AssertionError:
                print('Failed to open %s' % s)
                self.cur_pos+=1
                continue
            w = int(self.cap[curr_batch].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap[curr_batch].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap[curr_batch].get(cv2.CAP_PROP_FPS) % 100
            frames = int(self.cap[curr_batch].get(cv2.CAP_PROP_FRAME_COUNT))
            _, self.data[i] = self.cap[curr_batch].read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, self.cap[curr_batch], self.cur_pos+1]), daemon=True)
            print(' success (%gx%g at %.2f FPS having %g frames).' % (w, h, fps, frames))
            curr_batch+=1
            self.cur_pos+=1
            thread.start()
            print('')  # newline

        if all( v is None for v in self.data ):
            return
        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.data], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap, cur_pos):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.data[index] = cap.retrieve()
                if self.def_img_size is None:
                    self.def_img_size = self.data[index].shape
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.data.copy()
        img = []

        for i, x in enumerate(img0):
            if x is not None:
                img.append(letterbox(x, new_shape=self.img_size, auto=self.rect)[0])
            else:
                if self.cur_pos == self.n:
                    if all( v is None for v in img0 ):
                        cv2.destroyAllWindows()
                        raise StopIteration
                    else:
                        img0[i] = np.zeros(self.def_img_size)
                        img.append(letterbox(img0[i], new_shape=self.img_size, auto=self.rect)[0])
                else:
                    print('%g/%g: %s... ' % (self.cur_pos+1, self.n, self.sources[self.cur_pos]), end='')
                    self.cap[i] = cv2.VideoCapture(self.sources[self.cur_pos])
                    fldr_end_flg = 0
                    while not self.cap[i].isOpened():
                        print('Failed to open %s' % self.sources[self.cur_pos])
                        self.cur_pos+=1
                        if self.cur_pos == self.n:
                            img0[i] = np.zeros(self.def_img_size)
                            img.append(letterbox(img0[i], new_shape=self.img_size, auto=self.rect)[0])
                            fldr_end_flg = 1
                            break
                        self.cap[i] = cv2.VideoCapture(self.sources[self.cur_pos])
                    if fldr_end_flg:
                        continue
                    w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.cap.get(cv2.CAP_PROP_FPS) % 100
                    frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    _, self.data[i] = self.cap[i].read()  # guarantee first frame
                    img0[i] = self.data[i]
                    img.append(letterbox(self.data[i], new_shape=self.img_size, auto=self.rect)[0])
                    thread = Thread(target=self.update, args=([i, self.cap[i], self.cur_pos+1]), daemon=True)
                    print(' success (%gx%g at %.2f FPS having %g frames).' % (w, h, fps, frames))
                    self.cur_pos+=1
                    thread.start()
                    print('')  # newline

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0
