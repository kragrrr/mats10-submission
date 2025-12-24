# Cross-Layer Transcoder: Predicting Late-Layer Features from Early Embeddings

## Project Overview

This project investigates the relationship between early-layer embeddings and late-layer features in large language models by training a **transcoder** (affine probe) that maps from an LLM's embedding representation (first layer) to features present at a late layer of the residual stream.

### Research Question

Can we predict late-layer SAE (Sparse Autoencoder) features from early-layer embeddings? This helps us understand:
- How much of late-layer representations are predictable from input embeddings (dataset geometry)
- How much emerges from model computations (emergent transformations)
- Whether SAE features capture dataset properties or learned model computations

## Methodology

### Model and Data
- **Model**: Gemma 2 2B (Google DeepMind)
- **SAE Source**: Gemma Scope SAEs trained on layer 20 residual stream
- **Source Layer**: Layer 0 (embeddings, 2304 dimensions)
- **Target Layer**: Layer 20 (late layer, 16k SAE features)

### Pipeline

1. **Data Collection**
   - Extract paired activations: layer 0 embeddings and layer 20 SAE features
   - Collect ~15,000 samples from diverse text data
   - Balance dataset for active/inactive feature states

2. **Transcoder Training**
   - Train affine probes (linear transformations) to predict SAE features from embeddings
   - Evaluate using R² score and classification accuracy
   - Test multiple SAE features to find highly predictable ones

3. **Residual Analysis**
   - Train a cross-layer transcoder to predict layer 20 activations from layer 0
   - Compute residuals: `residual = layer20 - transcoder(layer0)`
   - Separate dataset geometry (predictable) from emergent computations (unpredictable)

4. **SAE Comparison**
   - Train baseline SAE on raw layer 20 activations
   - Train residual SAE on emergent-only activations (residuals)
   - Compare feature predictability between baseline and residual SAEs

## Key Findings

### Feature Predictability
- **Baseline SAE Features**: Mean accuracy ~75.5% (highly predictable from embeddings)
- **Residual SAE Features**: Mean accuracy ~74.2% (less predictable)
- **R² Score**: ~0.50 (50% of layer 20 variance predictable from layer 0)

### Interpretation
1. **Dataset Geometry**: ~50% of layer 20 activations are predictable from embeddings
2. **Emergent Computations**: ~50% emerge from model transformations
3. **SAE Features**: Baseline SAE features are more predictable, suggesting they capture dataset-level structure
4. **Residual Features**: Residual SAE features capture more emergent, model-learned structure

## Results Summary

```
✓ Cross-layer transcoder: R² = 0.5025 (50.2% predictable)
✓ Dataset geometry: 50.2% of layer 20
✓ Emergent computations: 49.8% of layer 20
✓ Baseline SAE: Mean accuracy 0.7555
✓ Residual SAE: Mean accuracy 0.7423
```

## Technical Details

### Architecture
- **Transcoder**: Simple affine transformation `y = Wx + b`
- **Probe**: Linear layer mapping 2304-dim embeddings → SAE feature activations
- **SAE**: JumpReLU Sparse Autoencoder (16k features)

### Training
- **Optimizer**: Adam
- **Learning Rate**: 1e-4 (tuned via grid search)
- **Loss Function**: MSE for regression, BCE for classification
- **Evaluation**: R² score, accuracy, confusion matrices

### Libraries
- PyTorch
- SAELens
- Transformers (HuggingFace)
- scikit-learn
- NumPy, Matplotlib

## Files

- `Tutorial_Gemma_Scope_from_Scratch.ipynb`: Main notebook containing the complete pipeline
- `README.md`: This file

## Future Work

1. Test on larger models (Gemma 2 9B)
2. Analyze intermediate layers to understand when features emerge
3. Investigate which types of features are most/least predictable
4. Compare different probe architectures (non-linear, deeper networks)
5. Study feature interpretability differences between baseline and residual SAEs

## Citation

If you use this work, please cite:
- Gemma Scope: [Technical Report](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf)
- SAELens: [Documentation](https://jbloomaus.github.io/SAELens/)

## License

This project is for research purposes. Please refer to the original model and SAE licenses (Gemma 2, Gemma Scope).

