# Copyright (c) 2022-2023 Faheem Sheikh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Python3 program classifying Thaats in Qawwali songs using neural networks

import argparse
import logging
from enum import Enum

from qdetect import QDetect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# Types of Thaats used in Indian/Pakistani
# classical music, reference: Noorang-e-Moosiqi By Urdu Science Board
# Matches the listing on: http://www.tanarang.com/english/glossary_eng.htm#Thaat
class Thaats(Enum):
    Asavari = 1,
    Bhairav = 2,
    Bhairavi = 3,
    Bilawal = 4,
    Kafi = 5,
    Kalyan = 6,
    Khamaj = 7,
    Marwa = 8,
    Poorbi = 9,
    Todi = 10,
    Unknown = 11

# Derived class from Qawwali Detector to reuse some data-preprocessing
# functionality. The child class overrides feature extraction and classification
# methods
class ThaatDetect(QDetect):
    def __init__(self, qDir):
        logger.info("Thaat Detector initialized with Qawwali songs from {}".format(qDir))
        # Call parent object assuming the data has been never been loaded
        super().__init__(songDir=qDir, reload=True)

    def decompose(self):
        logger.info("Thaat Detector feature decomposer")

    def classify(self):
        logger.info("Thaat Classifier")

if __name__ == '__main__':
    qParser = argparse.ArgumentParser(description="Qawwali thaat classification program")
    qParser.add_argument("songs_dir", type=str, help="folder/directory containing songs to be evaluated")

    qArgs = qParser.parse_args()

    thaatD = ThaatDetect(qArgs.songs_dir)
    thaatD.decompose()
    thaatD.classify()

