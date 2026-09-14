# Fall Detection

Trains an LSTM to continuously monitor a simulated 3-axis accelerometer stream and flag a fall -- a multi-phase event (free-fall dip, impact spike, then a long stretch of post-fall stillness) spread across a much longer sequence than the gesture example's fixed windows, with a per-timestep label rather than one label for the whole window. No dataset to download -- every sequence is synthesized on the fly.

## In an embedded system

This is the model shape behind a real deployed product category: fall-detection pendants and smartwatch fall alerts for elderly or at-risk users. The device is always on, always watching a live accelerometer stream, and has to decide -- continuously, on-device, on a battery-powered microcontroller -- whether what just happened was a genuine fall worth raising an alarm for.

That "genuine" is the hard part, and it's why this example uses an LSTM rather than the plain RNN the gesture example uses. A fall isn't one instant, it's a *pattern over time*: a brief free-fall dip, then a sharp impact, then -- critically -- the person doesn't get up normally afterward. Confirming a real fall (and not, say, just setting the device down hard on a table) means remembering the free-fall dip once the impact arrives, and then continuing to watch for unusual stillness for a long stretch afterward. A plain RNN's hidden state tends to wash out a brief early signal over that many quiet timesteps (every step forces it back through a squashing nonlinearity); an LSTM's gated cell state can hold onto it instead. See the top-level README's LSTM section, and the discussion in this repo's commit history, for the fuller "why LSTM over RNN" argument.

## Model architecture

![LSTM architecture: Input 3 accelerometer axes, into a 16-unit gated LSTM layer with two labeled recurrent self-loops -- cell state c (gated forget/input) and hidden state h (previous timestep) -- into a single sigmoid output for fall probability; captioned "one timestep per call"](architecture.svg)

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 3 | one accelerometer sample (x, y, z) per call |
| 1 | LSTM | 16 | fixed gate nonlinearities (sigmoid ×3, tanh); carries **two** persistent states across calls -- the hidden state `h` (like RNN) and the cell state `c` (LSTM-specific, see `LAYER_TYPE_LSTM`'s comment in `nn.h`) |
| 2 | Output | 1 | Sigmoid -- fall probability at this timestep |

1,297 trainable parameters -- still small (under 6KB as 32-bit floats), just over 3x the gesture example's RNN, since each of the four gates (input, forget, cell-candidate, output) has its own weight row per hidden unit. `train.c` calls `nn_train()`/`nn_predict()` once per timestep of a 150-sample monitoring window, with the label itself varying per timestep (0 throughout normal activity, flipping to 1 at the fall's onset and staying 1 through the end of the window) rather than one label for the whole window the way the gesture example does. `nn_reset_state()` (which zeros *both* `h` and `c` for an LSTM layer) is called before each new window.

## Build and run

```
cd examples/fall_detection
make
```

Train (synthesizes fresh training/validation sequences on the fly -- nothing to download):
```
./train fall_model.txt
```

Evaluate against a fresh, larger, independently-generated batch and print per-timestep accuracy, recall, false alarms, and detection latency:
```
./test fall_model.txt
```

## Sample output

`train.c` first prints one example sequence of each kind (every 3rd timestep shown, condensed to acceleration magnitude + label), so the free-fall dip and impact are visible before any training happens. Excerpt of the FALL example, around the onset (a normal sequence stays at label 0 and |a|≈1g throughout):
```
Sample FALL sequence (|accel| in g, every 3rd timestep):
  ...
  t= 57: |a|=0.98  label=0
  t= 60: |a|=1.06  label=0
  t= 63: |a|=1.00  label=0
  t= 66: |a|=0.04  label=1      <- free-fall: magnitude collapses toward 0
  t= 69: |a|=0.02  label=1
  t= 72: |a|=0.02  label=1
  t= 75: |a|=1.00  label=1      <- impact happened between t=72 and t=75 (not sampled at this stride)
  t= 78: |a|=1.01  label=1      <- post-fall stillness: label stays 1 for the rest of the window
  ...
```

Training log:
```
train error, validation error, learning rate
0.01934, 0.09307, 0.05000
0.00688, 0.01075, 0.05000
0.00469, 0.00549, 0.05000
0.00194, 0.00533, 0.05000
0.00149, 0.00340, 0.05000
0.00125, 0.00183, 0.05000
0.00355, 0.00925, 0.05000
0.00092, 0.00948, 0.05000
0.00383, 0.00687, 0.05000
0.00163, 0.00649, 0.05000
0.00074, 0.00459, 0.05000
No validation improvement for 5 epochs (best: 0.00183) -- stopping early.
Final (last epoch) train error: 0.000738, validation error: 0.004594
Best validation error (the model saved to disk): 0.001829
Training epochs: 11
Per-timestep accuracy: 2951/3000 = 98.37%
Falls detected: 10/10
False alarms (normal sequences that ever triggered): 0/10
Average detection latency (timesteps after true onset): 0.0
```

`test fall_model.txt` (a fresh, larger, independently-generated batch):
```
Test sequences: 50 normal, 50 fall (150 timesteps each, freshly synthesized)
Per-timestep accuracy: 14947/15000 = 99.65%
Falls detected (recall): 50/50 = 100.0%
False alarms (normal sequences that ever triggered): 0/50 = 0.0%
Average detection latency (timesteps after true onset): 0.0
```
