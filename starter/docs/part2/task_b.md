## Task B: supplied and combined Google decoders

### Input and target

Task B reads only `ml/ml_google_decoder_example.parquet` and evaluates the
supplied Google decoder predictions separately for code distances 3 and 5.

For each distance, the model input contains five features:

1. detector-event density: `detector_event_count / detector_count`;
2. belief matching prediction;
3. correlated matching prediction;
4. PyMatching prediction;
5. tensor-network contraction prediction.

The target is `actual_observable_flip`. The supplied `data_split` is used
unchanged: the combined model is fitted on the training split, its decision
threshold is selected on the validation split, and the test split is used once
for the reported metrics.

### Models

- **Prior baseline:** predicts the majority class estimated from the training
  split and uses the training flip rate as its probability.
- **Supplied decoders:** the four precomputed decoder predictions are evaluated
  directly on the same test examples. They are binary outputs, so Brier score
  is not applicable.
- **Combined decoder:** a logistic-regression model using detector-event
  density plus the four supplied decoder predictions.

The combined model is trained separately for distance 3 and distance 5. Its
classification threshold is selected on the validation split from the same
fixed threshold grid used by the implementation.

### Results

<!-- results:task_b -->

### Decoder agreement and complementarity

The four supplied decoders are also compared by the overlap of their test-set
errors. Pairwise error overlap is measured using the Jaccard index on the sets
of wrongly predicted shots. This shows whether the decoders mostly fail on the
same shots or whether they make complementary mistakes that a combined model
could exploit.

### Limitations

- Distance 3 and distance 5 are modelled separately and are never merged into
  one detector vector.
- The combined model is linear, so it can only exploit additive relationships
  between detector-event density and the four supplied predictions.
- The supplied decoders are evaluated from their stored binary predictions;
  their original training and inference times are therefore not measured here.