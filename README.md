# Note:

Phd Thesis on Vehicle Re-Identification for Biplav

## Data

VERI WILD = 40671 identities.

# Batch size

P = 8
K = 4
Batch_size = P \* K = 32

Training and Validation split = 80:20

No. of training iterations = Total no. of identities in training set / P = 5083

# Model Architecture

Custom Model with ResNet50 as backbone

# Loss Function

Triplet Loss
