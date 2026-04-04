import os, glob
import cv2

img_dir = r"case/sf"          # 改成你的图片文件夹
out_path = "out.mp4"
fps = 1

paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
paths = [p for p in paths if os.path.splitext(p)[1].lower() in [".png", ".jpg", ".jpeg"]]
assert paths, "No images found"

first = cv2.imread(paths[0])
h, w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for p in paths:
    im = cv2.imread(p)
    if im is None:
        continue
    if im.shape[:2] != (h, w):
        im = cv2.resize(im, (w, h))
    vw.write(im)

vw.release()
print("saved:", out_path)
