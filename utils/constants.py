import os


databank_path = os.environ.get("databank_path", "E:\\Projects\\2.Plant_classifier\\data\\databank.npz")
model_path = os.environ.get("model_path", "E:\\Projects\\2.Plant_classifier\\srcs\\plant_classifier_api\\models\\model_2021-10-06-23-39_acc_0.6_step_86328.pth")

predict_size = eval(os.environ.get("predict_size", "(224, 224)"))
strange_threshold = float(os.environ.get("strange_threshold", "1.2"))
