# Source code for plant classifier API endpoint

Training model codes can be found at [Training repo](https://github.com/BuiKhoi/large_scale_plant_classifier)

Most of the source codes in this repo is copied from the training repo. So take a look if you need any futher information about training processes


# Running tutorial
You will have to install the required packages with:
```
pip install -r requirements.txt
```

Download sample model at [Google Drive](https://drive.google.com/file/d/1yYY7QoCSTkJvsJrsCHR7sUwrBRzv6-lO/view?usp=sharing), put it somewhere on your computer, remember its file path

Download sample databank at [Google Drive](https://drive.google.com/file/d/1-5IwFMy8Y3tT3-0ZfTRrXTd-pBS8RSSu/view?usp=sharing), also remember its path

Replace the two paths in [constant.py](utils/constants.py)

Sample testing data can be downloaded from [Google Drive](https://drive.google.com/file/d/16yM6-4KlMpiGeH4G_iXluZ4FRR3PPulB/view?usp=sharing), unzip them.

Then start the api with:
```bash
uvicorn main:app
```

For more starting arguments of the api, please checkout [Uvicorn's Documentation](https://www.uvicorn.org/).

After the application is started, you can test the API from `localhost:8000/docs`. Address may differ if you change uvicorn's start arguments
