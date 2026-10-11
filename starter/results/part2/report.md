# Part II: AI/ML consumer of the QEC pipeline

Run `make train` from `starter/` in the course workspace. It reads only the
two ML input tables, `ml/ml_syndrome_decoder_example.parquet` and
`ml/ml_google_decoder_example.parquet`, never Gold, Silver, or the source
archives. One run writes the fitted model files `predictions.parquet`,
`metrics.json`, `run.json` and this report to `results/part2/`. The tables
below are generated from `metrics.json` in the same run.

All tasks use the supplied splits. Models are fitted on the training split,
settings and thresholds are chosen on the validation split and the test split
is used once for the reported metrics. Within a task, every model is compared
on identical test examples.

**Reading the tables.** All values are on the test split. The logical-error
rate is the share of test examples whose flip is predicted wrongly. Training
time covers only fitting on the training split and prediction time only
predicting the test split, both in seconds and without data loading or
feature preparation. `n/a` means not applicable.

## Run facts

- Data release: quantum-data-core version 3 (2026-08-05)
- Random seed: 42
- Code revision (source_tree_sha256): `02fdc1b53f68`
- ML input hashes (SHA-256):
  - `ml/ml_google_decoder_example.parquet`: `e8332a2d979d`
  - `ml/ml_syndrome_decoder_example.parquet`: `05abd4a78717`
- Dependencies: scikit-learn 1.9.0, numpy 2.5.1, scipy 1.18.0, joblib 1.5.3, threadpoolctl 3.6.0, pandas 2.3.1, pyarrow 20.0.0

Full values, feature order, split rules, settings, and timings are in `run.json`.

## Task A: weighted syndrome decoder

### Input and target

Task A reads only `ml/ml_syndrome_decoder_example.parquet`. Its 75,598 rows
represent 70,000,000 simulated shots: each row is one distinct syndrome and
label within one physical-fault-rate experiment, `sample_weight` is the
number of shots it stands for.`syndrome_model_input` turns each
row into 16 binary inputs ordered by round and then check (4 rounds × 4
checks), A 1 means that check changed compared with the previous round. The
target is `logical_error_label`: whether the logical value was flipped.

The supplied split is by fault rate, so the three partitions are different
noise levels:

| Split | Fault rate(s) | Rows | Shots | Shots with a flip |
| --- | --- | ---: | ---: | ---: |
| train | 0.00001, 0.00005, 0.0001, 0.001, 0.01 | 53,304 | 50,000,000 | 4.3 % |
| validation | 0.0005 | 1,407 | 10,000,000 | 1.1 % |
| test | 0.005 | 20,887 | 10,000,000 | 10.4 % |

The model is therefore tested on a noise level it did not see during
training.

### Why the weights matter

Rows and shots are very different here. The all-zero syndrome is about 83 % of all training shots but only 7 rows,
while the noisiest file (fault rate 0.01) supplies 93 % of the training rows
but only 20 % of the training shots. Fitting or scoring without the weights would
mostly describe that one specific noisy file. Both models are therefore fitted with
`sample_weight`, and every reported metric (logical-error rate, balanced
accuracy, and Brier score) is weighted by it.

### Why these models

- **Weighted prior baseline:** a `DummyClassifier(strategy="prior")` fitted
  with the weights. Most shots do not flip, so it always predicts "no flip"
  and gives the training flip rate as its probability. It is the bar any
  decoder has to clear.
- **Weighted logistic regression:** the simplest model that uses all 16 bits.
  It learns one weight per bit and adds them up, which makes it fast,
  repeatable (fixed seed), and easy to inspect.

The decision threshold is chosen on the validation split only, from a grid of
0.05 to 0.90 in steps of 0.05 (the lowest weighted validation error wins) and
is then applied once to the test split. The chosen threshold, feature order,
split rule, seed, and timings are recorded in `run.json`.

### Results

| Model | Distance | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_prior` | n/a | 0.1040 | 0.5000 | 0.0969 | 0.0120 | 0.0003 | n/a |
| `weighted_logistic_regression` | n/a | 0.1042 | 0.5015 | 0.0910 | 0.3081 | 0.0013 | 0.90 |

Test split. Rates, balanced accuracy, and Brier score are weighted by `sample_weight`.

Logistic regression does not beat the baseline on the test split. Its
weighted logical-error rate is essentially the baseline's and its balanced
accuracy is close to 0.5. The validation search ends at the top of the
threshold grid, so the model only predicts a flip when it is very confident.
On the test split it almost always predicts "no flip", just like the
baseline. Its probabilities do carry some information: its Brier score is
lower than the baseline's, so it ranks risky syndromes higher. That ranking is
not strong enough to turn into better yes/no decisions.

The weak result is not caused by uninformative inputs. Even a perfect lookup
table over the 16 bits would be wrong on only about 1.8 % of the test shots
(the same syndrome can occur with both labels, so no decoder can reach 0 %).
That is far below the baseline's 10.4 %, so the bits contain a strong signal
that a linear model cannot use.

### What a weighted sum cannot represent

Whether the logical value flips depends on whether a check fires an odd or
even number of times, not on how many events there are in total. On the
training split:

| Check 0 changes in rounds (all other bits 0) | Shots with a flip |
| --- | ---: |
| 1 | 97 % |
| 1 and 2 | 0.1 % |
| 1, 2 and 3 | 96 % |
| 1, 2, 3 and 4 | 0.4 % |

One event usually means a real error that flips the logical value. Two events
on the same check in neighbouring rounds usually mean one wrong readout,
which changes the check and then changes it back, so nothing flips. Checks 0
and 2 behave this way (a single event flips the label about 97 % of the
time), while checks 1 and 3 rarely affect the label (2–4 %), presumably
because of where they sit relative to the logical observable.

Logistic regression adds one weight per bit. If one event pushes the score
towards "flip", two events push it further towards "flip", never back. It
cannot express "one event or the other, but not both" (an exclusive-or). The
fitted coefficients agree with this: checks 0 and 2 get large positive
weights in every round and checks 1 and 3 get much smaller ones, so the model
learns which checks matter but not the odd/even rule. A model that receives
the per-check parity or that can combine bits non-linearly would be the
natural next step, but it was not required here so it was not built.

### Discarded information and limitations

- The physical fault rate and experiment identity are not model inputs.
  The model cannot adapt to the test noise level, and the
  threshold is tuned on a quieter regime (1.1 % flips) than the one it is
  tested on (10.4 %).
- One linear model with default regularisation and a coarse threshold grid,
  which means that the result says nothing about what a non-linear decoder
  could achieve.
- Logistic-regression probabilities are better than the baseline's but still
  not calibrated for the test noise level, so they should not be read as
  exact flip probabilities.
- Every prediction keeps the row's `example_id`, so it can be traced through
  `gold.syndrome_ml_source` to the Gold observation and its Bronze source row.

## Task B: supplied and combined Google decoders

### Input and target

Task B reads only `ml/ml_google_decoder_example.parquet` and evaluates the
supplied Google decoder predictions separately for code distances 3 and 5.

For each distance, the combined model uses five features:

1. detector-event density: `detector_event_count / detector_count`;
2. belief matching prediction;
3. correlated matching prediction;
4. PyMatching prediction;
5. tensor-network contraction prediction.

The target is `actual_observable_flip`. The supplied `data_split` is used
unchanged. The combined model is fitted on the training split, its decision
threshold is selected using validation data only, and the final metrics are
computed once on the test split.

Distance 3 contains 100,000 test shots and distance 5 contains 25,000 test
shots.

### Models

The **prior baseline** ignores the decoder outputs and predicts the majority
class estimated from the training split. Its training flip rate is used as the
predicted probability.

The four **supplied decoders** are evaluated directly on exactly the same test
shots. Their outputs in the ML table are binary predictions rather than
probabilities, so Brier score is not applicable. Their original training and
inference times are also unavailable because the supplied predictions were
computed before Part II.

The **combined decoder** is a logistic-regression model fitted separately for
distance 3 and distance 5. It uses detector-event density together with the
four supplied decoder predictions. The classification threshold is selected
on the validation split from a fixed 0.05–0.95 grid.

### Results

| Model | Distance | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_prior_baseline` | 3 | 0.4929 | 0.5000 | 0.2500 | 0.0001 | <0.0001 | 0.50 |
| `B_belief_matching` | 3 | 0.4013 | 0.5985 | n/a | n/a | n/a | n/a |
| `B_correlated_matching` | 3 | 0.4220 | 0.5779 | n/a | n/a | n/a | n/a |
| `B_pymatching` | 3 | 0.4364 | 0.5635 | n/a | n/a | n/a | n/a |
| `B_tensor_network` | 3 | 0.3967 | 0.6031 | n/a | n/a | n/a | n/a |
| `B_combined_logistic` | 3 | 0.3977 | 0.6019 | 0.2352 | 0.0746 | 0.0053 | 0.51 |
| `B_prior_baseline` | 5 | 0.5015 | 0.5000 | 0.2500 | <0.0001 | <0.0001 | 0.50 |
| `B_belief_matching` | 5 | 0.3994 | 0.6006 | n/a | n/a | n/a | n/a |
| `B_correlated_matching` | 5 | 0.4226 | 0.5774 | n/a | n/a | n/a | n/a |
| `B_pymatching` | 5 | 0.4548 | 0.5452 | n/a | n/a | n/a | n/a |
| `B_tensor_network` | 5 | 0.3952 | 0.6048 | n/a | n/a | n/a | n/a |
| `B_combined_logistic` | 5 | 0.3977 | 0.6026 | 0.2341 | 0.0202 | 0.0007 | 0.55 |

Test split.

For distance 3, the prior baseline has a logical-error rate of 0.49295. All
four supplied decoders improve substantially on this baseline. The
tensor-network decoder performs best with an error rate of 0.39668, followed
by belief matching at 0.40133, correlated matching at 0.42197, and PyMatching
at 0.43636.

The combined logistic-regression decoder reaches a logical-error rate of
0.39768 with balanced accuracy 0.60189. Its validation-selected threshold is
0.51. The combined model therefore performs almost as well as the strongest
supplied decoder, but does not improve on the tensor-network result.

For distance 5, the prior baseline is again close to chance with an error rate
of 0.50152. The tensor-network decoder is again strongest at 0.39524, followed
by belief matching at 0.39944, correlated matching at 0.42260, and PyMatching
at 0.45480.

The combined model again reaches an error rate of 0.39768, with balanced
accuracy 0.60265 and a validation-selected threshold of 0.55. It therefore
clearly improves on the prior baseline but remains slightly worse than the
tensor-network decoder.

The combined model also produces probabilities. Its Brier scores are 0.23515
for distance 3 and 0.23406 for distance 5, compared with approximately 0.25
for the prior baseline.

### Decoder agreement and complementarity

The supplied decoders make related but not identical mistakes. For distance 3,
pairwise Jaccard overlap between decoder error sets ranges from approximately
0.36 to 0.53. For distance 5, it ranges from approximately 0.33 to 0.49.

At distance 3, all four decoders are wrong on 14,440 of the 100,000 test
shots, while at least one decoder is wrong on 69,824 shots. At distance 5,
all four are wrong on 2,713 of 25,000 test shots, while at least one decoder is
wrong on 18,908 shots.

The moderate pairwise overlap shows that the decoders contain some
complementary information rather than failing on exactly the same shots.
However, the simple linear combined model does not turn that complementarity
into an improvement over the strongest individual decoder. This suggests that
binary decoder outputs and detector-event density alone do not provide enough
information for a linear combiner to consistently improve on the
tensor-network prediction.

### Limitations

- Distance 3 and distance 5 are modelled separately and are never merged into
  one detector vector.
- The combined model is linear and can therefore exploit only relatively
  simple relationships between detector-event density and the four supplied
  predictions.
- The supplied decoders are available only as binary predictions. Their
  probability estimates, training procedures, and execution times are not
  available in the ML table.
- Performance is reported on the fixed supplied test split; threshold selection
  uses validation data only.

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

| Model | Distance | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `C_mlp` | 3 | 0.5002 | 0.4998 | 0.2558 | 12.04 | 0.0230 | 0.71 |
| `C_majority` | 3 | 0.5001 | 0.5000 | 0.2500 | 0.0030 | <0.0001 | 0.50 |

Test split.

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
