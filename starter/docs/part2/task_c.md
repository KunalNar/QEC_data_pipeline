## Task C: raw-detector MLP prototype

### Input and target

Task C reads only `ml/ml_google_decoder_example.parquet` and keeps the
fixed subset `distance = 3 AND shot_index < 12_500`: four distance-three
experiments × 12,500 shots = 50,000 shots. The supplied split gives 20,000
train, 5,000 validation, and 25,000 test shots (5,000 / 1,250 / 6,250 per
experiment), and the stored `data_split` agrees with `google_data_split` for
every row.

Each shot's 25 packed bytes are unpacked with the supplied
`unpack_little_endian_bits` into 200 binary inputs. A detector
bit is 1 when a stabilizer check changed between rounds. The target is
`actual_observable_flip`: whether the protected logical value was flipped at
the end of the 25 rounds. Labels are balanced, about 50 % flips in every
split.

Before training, the stage checks that every row has 200 detectors, that the
number of unpacked 1-bits equals `detector_event_count` for every shot, and
that all three splits are present and contain no other values. A unit test
fixes the bit order by packing detectors 0, 9, and 199 and checking that
exactly those columns are set.

### Why this model

The MLP is the required check that the pipeline can serve a model consuming
raw packed bits. Whether a logical flip occurred depends on combinations of
detector events, not on single bits, so a model that can combine inputs
non-linearly is the simplest reasonable choice. We used one small
scikit-learn `MLPClassifier` with 64 hidden units, a fixed random seed, and
L2 regularisation `alpha = 1.0`.

The settings were chosen on validation data only. With the scikit-learn
default regularisation the network reached zero training error while its
validation error stayed close to 0.5: it memorised the 20,000 training shots.
Stronger regularisation reduced the memorisation, but no setting we tried
(including a smaller network and a logistic regression on the same bits)
moved validation error clearly away from 0.5. We therefore kept one fixed,
moderately regularised network rather than tuning further.

The decision threshold is chosen on the validation shots from a 0.05–0.95
grid (the lower the error the better) and then applied once to the test shots.
A majority baseline, which ignores the detector bits and predicts the training flip rate, is scored on exactly the same test shots. The exact settings, iteration count, chosen threshold, and
timings of each run are recorded in `run.json`.

### Results

<!-- results:task_c -->

The MLP performs at chance level. Its test logical-error rate and balanced
accuracy are within noise of the majority baseline, and its Brier score is
slightly worse than the baseline's: its probabilities lean away from 0.5 but
are not more often right, so they are overconfident. The threshold selected
on validation also moves with the random seed during development, which is
consistent with the network having learned no stable signal.

The weak result is not caused by empty inputs. On the same 25,000 test shots
the supplied decoders reach logical-error rates of about 0.40
(tensor-network contraction 0.400, belief matching 0.400, correlated matching
0.421, PyMatching 0.432), so the detector bits do carry information. The
ceiling is low because errors accumulate over 25 rounds: if each round were
independent, a final error rate of 0.40 would correspond to roughly 3 % per
round. The gap between 0.50 and 0.40 is the signal the MLP fails to
extract.

### What a flat 200-bit vector does not show

The 200 bits are not unrelated features. They are consistent with 25 rounds
× 8 detectors in round order: reshaping the per-detector firing rates to a
25 × 8 grid shows that in the first round, detectors 0–3 fire much less often
(0.04–0.07) than later rounds, and that detectors 1, 3, 4, and 6 fire less in
every round (about 0.11 against 0.16). The ML table does not state this layout explicitly,
and the MLP is not told it. A flat vector hides:

- **Time:** that bits 3, 11, 19, (every 8th bit, so the same position modulo 8)  are the same detector in consecutive
  rounds. A matching decoder pairs such events: the same detector firing in
  two neighbouring rounds usually indicates a measurement error, not a real
  data error.
- **Space:** which detectors are neighbours on the chip, so which events can
  be explained by one error between them.
- **Position:** the detector coordinates and the processor location
  (`center_row`, `center_col`), which are not inputs.

A fully connected network has no built-in notion of input order: shuffling
the 200 columns the same way for every shot gives it an equivalent learning
problem. It must therefore learn all of this structure from 20,000 examples. In practice it
memorises individual training shots instead. A model given the round ×
detector layout explicitly would be the natural next step.

### Discarded information and limitations

- Only distance 3 and shots below 12,500 are used, as required. Distance 5
  (600 bits), the other 150,000 shots, the four supplied decoder predictions,
  and the processor location are not model inputs.
- One small architecture with limited regularisation tuning, which means that the result says
  nothing about what a structured model could achieve.
- The MLP's probabilities are overconfident (Brier score above the
  baseline's), so they should not be read as calibrated flip probabilities.
- Every prediction keeps the shot's `example_id`, so it can be traced through
  `gold.google_ml_source` to the Gold shot and its Bronze source members.
