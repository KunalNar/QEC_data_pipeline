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

<!-- results:task_b -->

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