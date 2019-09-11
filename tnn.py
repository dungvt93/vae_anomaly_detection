import cv2
import numpy as np
import datetime
import os, glob

class Model:
  def __init__(self, template_filepaths):
    # templateフォルダにあるすべての画像が対象
    template_filepaths = glob.glob(os.path.join(template_filepaths, 'template', '*.png'))
    self.template_imgs = []
    for paths in template_filepaths:
      self.template_imgs.append((cv2.imread(paths, 0), int(os.path.basename(paths)[:-4])))

  def load_model(self, checkpoint_name):
    pass
  
  def close(self):
    pass

  def detect(self, base_img, threshold=0.8):

    # 返却値
    out_boxes = []
    out_scores = []
    out_clipped_img = []

    for template_img, base_x in self.template_imgs:
      h, w = template_img.shape

      # マッチング対象範囲の限定
      x_start = max(0, base_x - w)
      x_end = min(base_img.shape[1], base_x + w)

      # 画像のグレイ化
      base_img_gray = cv2.cvtColor(base_img[:, x_start:x_end, :], cv2.COLOR_BGR2GRAY)

      # テンプレートマッチング
      res = cv2.matchTemplate(base_img_gray, template_img, cv2.TM_CCOEFF_NORMED)
      res = np.array(res)

      # 各yで最高一致率となるxの配列を算出
      best_xs = np.argmax(res, axis=1)

      # 最良のxを確定
      best_x = int(np.median(best_xs)) # 中央値
      # count = np.bincount(best_xs) # 最頻値
      # best_x = np.argmax(count)

      # 確定したxのなかで最高一致率のyを探す
      best_y = np.argmax(res[:,best_x])

      # 何も検出されなければ次のテンプレートへ
      if res[best_y][best_x] < threshold:
        out_boxes.append(None)
        out_scores.append(0.0)
        out_clipped_img.append(None)
        continue

      # アプリケーション側で処理できるよう出力値を成形
      pt = (best_x + x_start, best_y)
      out_boxes.append([pt[1], pt[0], pt[1]+h, pt[0]+w])
      out_scores.append(res[best_y][best_x])
      out_clipped_img.append(base_img[pt[1]:pt[1]+h, pt[0]:pt[0]+w])

    return out_boxes, out_scores, out_clipped_img

if __name__ == '__main__':
  model = Model("../20190830_dataset/")
  img_paths = glob.glob('../20190830_dataset/K_190902/*.png')
  for img_path in img_paths:
    img = cv2.imread(img_path)
    _, _, results = model.detect(img, 0.8)
    if len(results) != 0 and results[0] is not None:
      cv2.imwrite(img_path.replace('../20190830_dataset/K_190902', '../20190830_dataset/test_K'), results[0])
    # cv2.imwrite(img_path.replace('201907300948', 'result2'), results[1])
