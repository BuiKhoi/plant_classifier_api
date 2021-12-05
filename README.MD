# Source code for plant classifier API endpoint

Training model codes can be found at [Training repo](https://github.com/BuiKhoi/large_scale_plant_classifier)

Most of the source codes in this repo is copied from the training repo. So take a look if you need any futher information about training processes


# Running tutorial
You will have to install the required packages with:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
Note: If your environment doesn't have any CUDA device installed, you may want not to install `pytorch`, but running on a CUDA environment is highly recommended

Download sample model at [Google Drive](https://drive.google.com/file/d/1yYY7QoCSTkJvsJrsCHR7sUwrBRzv6-lO/view?usp=sharing), put it somewhere on your computer, remember its file path

Download sample databank at [Google Drive](https://drive.google.com/file/d/1-5IwFMy8Y3tT3-0ZfTRrXTd-pBS8RSSu/view?usp=sharing), also remember its path

Replace the two paths in [constant.py](utils/constants.py). For more described configurations, refer to 
`API Configuration` section below.

Sample testing data can be downloaded from [Google Drive](https://drive.google.com/file/d/16yM6-4KlMpiGeH4G_iXluZ4FRR3PPulB/view?usp=sharing), unzip them.

Then start the api with:
```bash
uvicorn main:app
```

For more starting arguments of the api, please check out [Uvicorn's Documentation](https://www.uvicorn.org/).

After the application is started, you can test the API from `localhost:8000/docs`. Address may differ if you change uvicorn's start arguments


# API Description
Plant Classifier API is an Application Programming Interface which will do one job: Predict what class which belongs to any of the plants within its database.

Plant Classifier API contains the following endpoints:

## Class prediction
Class prediction is available at:
```
http://<your_host>:<your_port>/predict/predict_class
```
With `your_host` and `your_port` is configured when you start uvicorn at the previous section.
### Inputs
Class prediction API requires these inputs:
* file: ImageFile: This is an object of type file, which contains information about the image you're predicting.
Supported image extensions is `jpg, jpeg and png`.

### Output
The output will consist of two arrays, for example:
```
{
  "predicted_classes": [
    "7156",
    "272711",
    "81860",
    "59206",
    "288025"
  ],
  "similarities": [
    0.7975,
    0.2301,
    0.2085,
    0.1961,
    0.1958
  ]
}
```
With `predicted_classes` is the descending classes with has the most similarity to the input image, it can be any name or class index, which is decided when you create the databank. And `similarities` is the corresponding similarity for the classes.

# API Configuration
You will need to configure this API at needs. To do so, first check out [constant.py](utils/constants.py)
You can see 4 configurable options:
* databank_path: path to the databank, databank is your database which contains images embeddings
and their labels. The API will try to find the class with most similar embeddings base on the databank given.
To generate databank for your own, please refer to `Databank` section from [training repo](https://github.com/BuiKhoi/large_scale_plant_classifier) 
* model_path: path to the trained model. Note that the model using here and the model used to generate the databank
should be the same, otherwise the result may not correct.
* predict_size: Default 224x224. This is the predicting size. Images will be resize to this size no matter 
what size your original image is, so be careful. The bigger the image, the higher the accuracy will get but
the model will run slower. Note that if you want to change this `predict_size`, you have to also configure (and train)
the model to fit it.
* strange_threshold: Default 1.2. This is the threshold of similarity to be considered "strange", which cannot
be found similar to any result in the databank. Any image with predicted similarity exceed this value will result "Dunno" when predicting.
* similarity_limit: Default 5. You will define how many similar class you wish to return when call the API

Note: These variables are set as environment variable, so if you need to run as a docker container you will
only need to configure in docker-compose file

# API Sample and testing
Some samples and testing is also contained within this repo. You can check out at [Unittests](). Which
includes sample API calls and post-processing result
```bash
python unittest -m tests/unittests/class_predict_test.py
```
Note: You have to run unittest with the API running otherwise it will catch `HTTPConnectionError`