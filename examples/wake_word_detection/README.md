# Wake Word Detection

Trains a recurrent network to classify a whole ~1-second spoken utterance as a single wake word ("marvin", by default) or not -- the model shape behind "Hey Siri"/"Alexa"/"OK Google"-style always-listening voice assistants, scaled down to something that fits a microcontroller. Unlike [`gesture_recognition`](../gesture_recognition/README.md) and [`fall_detection`](../fall_detection/README.md), there's no synthetic data generator here: `prepare_data.c` builds `samples.csv` from the real [Google Speech Commands dataset](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) (CC BY 4.0), the way [`character_recognition`](../character_recognition/README.md) trains on real MNIST digits rather than synthesized ones.

`train.c` trains a single GRU-based network. That's the result of a real comparison, not an assumption -- see [Why GRU](#why-gru) below.

## In an embedded system

This is the always-on listener at the front of a voice-controlled device: a doorbell, a smart speaker, a hearing aid, anything that needs to react to being addressed without streaming raw audio to a server (or even waking up a bigger, power-hungrier speech-recognition model) just to find out nobody said anything. It runs continuously, on-device, on a power budget where "always transcribing everything" isn't an option -- its only job is deciding, from a rolling ~1-second window of its own microphone, whether the one word it cares about was just spoken.

Two pieces have to work together to make that possible, and this example is built to keep them visibly separate:

1. **Feature extraction** (`audio_features.[ch]`, at the repository root): turns a window of raw PCM samples into a short sequence of log-mel energy frames, normalized to zero mean/unit variance per clip -- the same transform a full speech-recognition pipeline uses, just feeding a much smaller downstream model. This is real signal-processing code, not a training-time-only shortcut, because it has to run identically on-device against the live mic buffer.
2. **The classifier** (this directory's `train.c`): a small GRU-based network that reads that feature sequence one frame at a time and outputs a single wake-word probability.

`predict.c` is the concrete illustration of that split: everything in it except loading the sample WAV file (`wav_reader.[ch]`, which a real target wouldn't need -- it already has PCM from its mic driver) is exactly what firmware would do.

## Why a real dataset, unlike the other RNN examples?

`gesture_recognition` and `fall_detection` synthesize their own training data because doing so costs nothing and the point being made (RNN/GRU/LSTM can hold information across timesteps) doesn't depend on realism. Feature extraction is different: there's no meaningful way to fake "what an FFT and mel filterbank do to real speech," so this example needs real audio, and `prepare_data.c` -- not a downloaded, already-flattened CSV the way MNIST's `samples.csv` is -- is the one place in this codebase that turns a raw WAV file into the same feature rows a live target would compute for itself. See `prepare_data.c`'s top comment for the full rationale.

The positive class is a single word, "marvin" by default -- one of two words ([Warden, 2018, §2](https://arxiv.org/abs/1804.03209)) the dataset's own authors deliberately included so they could double as a pretend wake word. The negative class mixes a balanced sample of the dataset's other spoken words with 1-second chunks cut from its `_background_noise_` recordings, so the model learns to stay quiet through both other speech and ambient noise/silence, not just one or the other. The wake word is a command-line argument to `prepare_data`, not hardcoded, so trying a different one is a rerun, not a code change.

## Why GRU?

Early versions of this example trained a plain RNN, a GRU, and an LSTM side by side -- the same three-way comparison [`fall_detection`](../fall_detection/README.md) runs on its own synthetic task. Unlike that comparison, where all three architectures reached the same ceiling, on this real, noisier task they didn't: measured on the held-out `test.csv` split (one comparison run, real data, not something regenerated fresh and identical every time the way the synthetic examples' numbers are):

| Architecture | Test accuracy | False-reject | False-accept |
|---|---|---|---|
| Plain RNN | ~81% | ~9% | ~29% |
| **GRU** | **~88%** | **~8%** | **~17%** |
| LSTM | ~77% | ~45% | ~4% |

GRU won clearly and consistently, so it's the only architecture kept here now -- unlike `fall_detection`, which keeps its three-way comparison because none of its architectures came out ahead on its task. `NN_OPTIMIZER_ADAM` was also tried on the plain RNN (it dramatically helped `fall_detection`'s own plain RNN on its synthetic task) and made things *worse* here at two different learning rates -- a good reminder that a fix measured on one task doesn't necessarily transfer to another, even for the same architecture. If you want to see any of this for yourself, the RNN/LSTM/Adam code no longer ships in this directory, but reconstructing it from `train.c` is a small, mechanical change (swap `LAYER_TYPE_GRU` for `LAYER_TYPE_RNN`/`LAYER_TYPE_LSTM`).

Two things mattered more than architecture choice, though, and both are worth knowing if you're adapting this example:

- **Per-clip feature normalization** (in `audio_features_extract_sequence()`) was the single biggest lever -- real recordings vary hugely in loudness and spectral tilt, and without normalizing that away, training was noisy and every architecture scored 15-20 points lower.
- **Learning rate**: the other RNN examples' plain-SGD rate of 0.05 was too large for this noisier, real-audio task and made validation error bounce instead of settle. A small sweep (0.003-0.03, two training runs each, since random init/shuffling isn't seeded and single-run test accuracy on this real dataset swings by several points) found 0.03 both the highest-scoring and the most consistent of the values tried (~87-89% test accuracy both runs, vs. lower and more erratic results at 0.01-0.02) -- see `train.c`'s tunable hyperparameters.

## Model architecture

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | `NUM_MEL_BINS` (one frame) | fed once per frame -- see `wake_word_data.h` |
| 1 | GRU | 32 | fixed gate nonlinearities; carries persistent state across the SEQUENCE_LEN frames of one clip |
| 2 | Output | 1 | Sigmoid -- wake-word probability |

`wake_word_data.h`'s defaults (16kHz, 512-sample/32ms frames, 320-sample/20ms hop, 20 mel bins) give `SEQUENCE_LEN` = 49 frames per 1-second clip. Every frame of one clip is trained against the SAME label -- the whole clip either is or isn't the wake word -- following [`gesture_recognition`'s](../gesture_recognition/README.md) "one label repeated every timestep" scheme rather than `fall_detection`'s per-timestep-varying one (see `train.c`'s top comment).

## Build and run

```
cd examples/wake_word_detection
make
```

The first `make` downloads and extracts the ~2.4GB Speech Commands archive, builds `prepare_data` and runs it to produce `samples.csv`, splits that into `train.csv`/`validation.csv`/`test.csv` via `../../split.py`, and builds `train`/`test`/`predict`. Expect the first run to take a while; `make clean` doesn't re-download or re-extract the dataset (only `make distclean` does).

Train the model:
```
./train wake_word_model.txt
```

Evaluate it against the held-out `test.csv` split and print accuracy, false-reject rate, and false-accept rate:
```
./test wake_word_model.txt
```

Run it against a single WAV file (e.g. a clip from `dataset/marvin/` or `dataset/<some other word>/`), the way an application would:
```
./predict wake_word_model.txt dataset/marvin/<some-file>.wav
```

To try a different wake word, rerun `prepare_data` directly and re-split, then train a fresh model against the new `train.csv`/`validation.csv`:
```
./prepare_data dataset samples.csv sheila
python ../../split.py samples.csv --train 0.8 --validation 0.1
./train wake_word_model_sheila.txt
```

## Sample output

`train.c` prints the same shape of training log as `fall_detection`/`character_recognition` -- a header line, one `train error, validation error, learning rate` row per epoch, early stopping once validation error stops improving, then a final summary:
```
train error, validation error, learning rate
...
No validation improvement for 5 epochs (best: <your run's number>) -- stopping early.
Final (last epoch) train error: <...>, validation error: <...>
Best validation error (the model saved to disk): <...>
Training epochs: <...>
Validation accuracy: <correct>/<total> = <...>%
False rejects (missed wake word): <...>/<...> = <...>%
False accepts (wrongly triggered): <...>/<...> = <...>%
```

`./test wake_word_model.txt` against the held-out test split:
```
Test clips: <total> (<wake word> wake word, <other> other), 49 frames each
Accuracy: <correct>/<total> = <...>%
False-reject rate (missed wake word): <...>/<...> = <...>%
False-accept rate (wrongly triggered): <...>/<...> = <...>%
```

The exact numbers vary run to run (random weight init and shuffling aren't seeded), but should land in the neighborhood of the [Why GRU](#why-gru) table above.
