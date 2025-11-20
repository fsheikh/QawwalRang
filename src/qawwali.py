### Qawwali recognition via computer vision AI
import argparse
from fastai.vision.all import *
from fastai.tabular.all import *
import dill
import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
# Needed in case logging has been previously setup on the system with callback handlers
logger.setLevel(logging.INFO)

DataDir = "./data"
ModelDir = "model"
# ModelName = "Resnet18Qawwali"
# ModelName = "Resnet18QawwaliLowFP"
ModelName = "Resnet18QRetrain"
ModelSuffix = ".pth"
LearnerSuffix = ".pkl"
FastAIInternalModel = False

def get_label(path):
  #logger.info(f"label being returned {path.parent.name}")
  return path.parent.name

def train_model(audioImagesArchive, train=False):
    audioImagesURI = audioImagesArchive.as_uri()
    audioImagesDir = untar_data(audioImagesURI)
    logger.info('Audio images files: {} extracted in directory: {}'.format(audioImagesURI, audioImagesDir))

    audioImages = get_image_files(audioImagesDir)
    count = 0
    for f in audioImages:
      if f.is_file():
        img = Image.open(f)
        count = count + 1

    logger.info(f'Number of audio images loaded: {count}')
    #sampleImg = Image.open(audioImages[69])
    #sampleImg
    #print(f"length of dataset={len(audioImages)} random item label={get_label(audioImages[60])}")

    splitter = RandomSplitter()
    try:
        train_idxs, valid_idxs = splitter(audioImages)
    except Exception as e:
      logger.error(f"Error during splitting {e}")

    logger.info(f"{len(train_idxs)} {len(valid_idxs)}")
    logger.info(f"Train files: {[audioImages[i] for i in train_idxs]}")
    logger.info(f"Validation files: {[audioImages[i] for i in valid_idxs]}")
    dls = ImageDataLoaders.from_path_func(Path(audioImagesDir),
                                          get_image_files(audioImagesDir),
                                          valid_pct=0.2,
                                          seed=42,
                                          batch_size = 4,
                                          label_func=get_label)

    train_ds = dls.train
    valid_ds = dls.valid

    logger.info(f"Training dataset size: {len(train_ds)}")
    logger.info(f"Validation dataset size: {len(valid_ds)}")
    logger.info(f"Train files: {train_ds}")
    logger.info(f"Validation files: {valid_ds}")

    model = vision_learner(dls, resnet18, pretrained=False,
                metrics=accuracy, model_dir=Path(ModelDir))

    logger.info(f"Number of model parameters = {len(list(model.parameters()))}")

    if train:
        model.fit_one_cycle(10, 1e-3)
        model.path = Path(os.getcwd()) / Path(ModelDir)
        logger.info(f"Saving Model at {model.path / Path(ModelName+LearnerSuffix)}")
        model.save(ModelName)
        model.export(ModelName, pickle_module=dill)
    else:
        logger.info(f"path={model.path} directory={model.model_dir}")
        # TODO: Parameterize to use the model from fastai internal path (need at least one training cycle)
        if FastAIInternalModel:
            model.load(Path(ModelName))
        else:
            # Load pre-trained model from a past training cycle (possibly on another machine)
            model.path = Path(os.getcwd())
            model.load(model.path / Path(ModelName))

    return model


if __name__ == '__main__':
    aParser = argparse.ArgumentParser(description="Qawwali inference via Resent")
    aParser.add_argument("feature_dir", type=str, help="Directory containing images with MFCC/CQT plots")
    aParser.add_argument("--reload", dest="reload", action="store_true", help="Reloads the neural network (needed to load the previous state)")
    aParser.add_argument("--retrain", dest="retrain", action="store_true", help="Retrain the neural network (default will just run inference)")

    aArgs = aParser.parse_args()

    featureDirPath = Path(aArgs.feature_dir)
    featureListing = list(featureDirPath.glob('**/*.png'))
    if aArgs.reload:
        # Pre-prepared audio images needed for recreating FastAI learn object
        audioImagesArchive = Path(os.path.abspath(DataDir + '/train/audio_images.tar.gz'))
        model = train_model(audioImagesArchive, False)
        model.show_results()
        for feature in featureListing:
            logger.info(f"Running inference on {feature}")
            fullDec, dec, modelPrediction = model.predict(feature)
            logger.info(f"Model output: {modelPrediction} Decoded prediction: {fullDec}")
    elif aArgs.retrain:
        # Pre-prepared audio images needed for training
        audioImagesArchive = Path(os.path.abspath(DataDir + '/train/audio_images.tar.gz'))
        model = train_model(audioImagesArchive, True)
        model.show_results()
        for feature in featureListing:
            logger.info(f"Running inference on {feature}")
            fullDec, dec, modelPrediction = model.predict(feature)
            logger.info(f"Model output: {modelPrediction} Decoded prediction: {fullDec}")
    else:
        # Run model prediction/inference
        # For independent inference to work, "export" the model and then load it
        # as a full learner object
        # https://jss367.github.io/saving-and-loading-models-in-fastai.html
        learnerPath = Path(os.getcwd()) / Path(ModelDir) / Path(ModelName+LearnerSuffix)
        logger.info(f"Loading model from previous export {learnerPath}")
        model = load_learner(learnerPath, cpu=True, pickle_module=dill)
        for feature in featureListing:
            logger.info(f"Running inference on {feature}")
            fullDec, dec, modelPrediction = model.predict(feature)
            logger.info(f"Model output: {modelPrediction} Decoded prediction: {fullDec}")

