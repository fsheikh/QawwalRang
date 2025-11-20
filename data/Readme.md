## Audio Features

Two set of time-varying audio features are considered
- Constant-Q transform a.k.a [CQT](https://en.wikipedia.org/wiki/Constant-Q_transform)
- Mel-Frequency Cepstral Coefficients a.k.a [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

These features are captured as function of time in two plots which act as a feature image for the ML algorithms, both rule-based as well as neural networks.

## Training and validation
Audio features used in training process: [tarball](https://drive.google.com/file/d/1rlgL3Io_HDycDjfnnN6H0-jbryeCUrsk/view?usp=drive_link). If you want to retrain the network you would to place this compressed tarball in `data/train` directory. The tarball contains *Qawwali* as well as *non-Qawwali* genre songs from Indian/Pakistani semi-classical music.

## Testing
Audio features used during testing from *non-Qawwali* genre: [tarball](https://drive.google.com/file/d/1CMvX2CHs1exkn0L-s5b4DH6aVraG0zPE/view?usp=sharing). Unpack this into `data/test` directory before testing.
