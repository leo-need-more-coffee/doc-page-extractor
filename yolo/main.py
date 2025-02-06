from doclayout_yolo import YOLOv10
from paddleocr import PaddleOCR, draw_ocr

def main():
  # Load the pre-trained model
  model = YOLOv10("/Users/taozeyu/codes/github.com/moksize91/test-yolo/models/doclayout_yolo_ft.pt")
  image_path = "/Users/taozeyu/codes/github.com/moksize91/test-yolo/sources/page1.png"

  # Perform prediction
  det_res = model.predict(
    image_path,   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold
    device="cpu"    # Device to use (e.g., 'cuda:0' or 'cpu')
  )
  print(det_res[0].__dict__['boxes'])

  ocr = PaddleOCR(use_angle_cls=True, lang='zh') # need to run only once to download and load model into memory
  img_path = image_path
  result = ocr.ocr(img_path, cls=True)
  for idx in range(len(result)):
    res = result[idx]
    for line in res:
      print(line)

  # Annotate and save the result
  # annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
  # cv2.imwrite("result.jpg", annotated_frame)

if __name__ == "__main__":
  main()