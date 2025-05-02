### Qawwali recognition via computer vision AI
from fastai.vision.all import *
import glob
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
# Needed in case logging has been previously setup on the system with callback handlers
logger.setLevel(logging.INFO)

DataDir = "./data"
ModelDir = "models"
ModelName = "Resnet18Qawwali"
ModelSuffix = ".pth"

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

        model.save(ModelName)
        logger.info(f"Model path: {model.path}/{model.model_dir}")
    else:
        model.load(Path(ModelName))

    return model



# Load pre-prepared audio images for loading into AI model
audioImagesArchive = Path(os.path.abspath(DataDir + '/train/audio_images.tar.gz'))
# Train a computer vision model
model = train_model(audioImagesArchive, True)
model.show_results()
# Run model prediction/inference
TestDir = DataDir + '/test/qawwali'
TestFeatures = ['BehadRamzaan.png', 'NamiDaanam.png', 'MahiyaTrimmed.png', 'GardishonKaayMaaray.png']
for feature in TestFeatures:
    featureImgPath = Path(TestDir) / Path(feature)
    logger.info(f"Running inference on {featureImgPath}")
    fullDec, dec, modelPrediction = model.predict(featureImgPath)
    logger.info(f"Model output: {modelPrediction} Decoded prediction: {fullDec}")

#learn.fine_tune(6)
#learn.show_results()
#learn.model = learn.model.cpu()

#xb,yb = learn.dls.one_batch()
#init_loss = learn.loss_func(learn.model(xb), yb)
#learn.fit(10)
#xb,yb = learn.dls.one_batch()
#final_loss = learn.loss_func(learn.model(xb), yb)
#assert final_loss < init_loss, (final_loss,init_loss)
