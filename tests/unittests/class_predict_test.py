import unittest
import os
import requests


cwd = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(cwd, "../data/")


class ClassPredictTest(unittest.TestCase):
    def test_01_test_predict_class(self):
        test_images = [
            {
                "file_path": os.path.join(test_folder, "216096.jpg"),
                "class": "206029"
            },
            {
                "file_path": os.path.join(test_folder, "216121.jpg"),
                "class": "Dunno"
            }
        ]

        for ti in test_images:
            image_ref = open(ti["file_path"], 'rb')
            response = requests.post("http://localhost:8000/predict/predict_class", files={
                "file": image_ref
            })

            response = response.json()
            self.assertEqual(response["image_class"], ti["class"])
            image_ref.close()
