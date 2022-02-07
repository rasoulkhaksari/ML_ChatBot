import settings
from ml import Training,Algorithms
from os import path
from tensorflow import keras
import xgboost as xgb
import pickle
import json
from bot import Bot
from gui import GUI
from log import Logger
LOG = Logger()

if __name__=='__main__':
    try:
        words,classes,model=None,None,None
        if path.exists(f"{settings.OUTPUT_PATH}/words.pkl") and path.exists(f"{settings.OUTPUT_PATH}/classes.pkl"):
            with open(f"{settings.OUTPUT_PATH}/words.pkl", "rb") as words_file:
                words = pickle.load(words_file)
            with open(f"{settings.OUTPUT_PATH}/classes.pkl", "rb") as classes_file:
                classes = pickle.load(classes_file)
            if Algorithms(settings.ALGORITHM)==Algorithms.XGBoost:
                model = xgb.XGBClassifier()
                model.load_model(f"{settings.OUTPUT_PATH}/model.json")
            else:
                model = keras.models.load_model(f"{settings.OUTPUT_PATH}/model.h5")
        else:
            training = Training(data_file=settings.DATA_FILE,output_path=settings.OUTPUT_PATH,algorithm=Algorithms(settings.ALGORITHM))
            (model,words,classes) = training.train()
        bot = Bot(model,words,classes,intents=json.load(open(settings.DATA_FILE)))
        gui = GUI(bot)
        gui.run()
    except Exception as exc:
        LOG.INFO('Error in training the model, check the log file for details.')
        LOG.ERROR(exc.args[0])