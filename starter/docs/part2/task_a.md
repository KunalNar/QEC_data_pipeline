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

<!-- results:task_a -->

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
