from imageai.Detection import ObjectDetection

model_path = "./models/yolo-tiny.h5"
input_path = "./input/test_im.jpg"
output_path = "./output/newimage.jpg"

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
