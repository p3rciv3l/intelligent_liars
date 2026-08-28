# Use a hierarchical sparse direction search

Routine Optuna explores every qualified source layer, writer-layer region, direction family, writer policy, and projection/reflection region through a hierarchical sparse parameterization rather than assigning an unrestricted coefficient to every stored vector. Recipes select general, domain-specific, intermediate, or mixed candidates, construct a bounded-rank QR or SVD basis, and tune independent attention and MLP writer kernels through strength two; guaranteed coverage prevents sparsity from silently excluding a layer or direction family.
