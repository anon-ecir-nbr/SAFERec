# SAFERec
### Code implementation for submission: "Self Attention and Frequency Enriched Model for the next-basket recommendation"
### At this time, this repository contains only the model architecture code, without the training loop, data preprocessing, and other related components.

# Optimal Hyper Parameneters for SAFERec
| **Dataset** | **# heads** | **# layers** | **d** | **F<sub>max</sub>** | **L** |
|-------------|-------------|--------------|-------|----------------------|--------|
| TaFeng      | 2           | 4            | 64    | 47                   | 256    |
| Dunnhumby   | 4           | 4            | 64    | 36                   | 32     |
| TaoBao      | 4           | 2            | 128   | 5                    | 128    |



# Hyper Parameneters Search Space for Optuna
## P-pop
Prepcoessing: categorical: [None, "binary", "log"]

## GP-Pop
MinFreq: int: min_freq $\in [1,20]$  
PreprocessingPopular: categorical: [None, "binary", "log"]  
PreprocessingPersonal: categorical: [None, "binary", "log"]  

## TIFU-KNN
NumNearestNeighbors: ctaegorical: [100, 300, 500, 700, 900, 1100, 1300]  
WithinDecayRate: categorical: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
GroupDecayRate: categorical: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
$\alpha$: categorical: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
GroupSizeDays: int:  group_size_days $\in [1, 365]$  
UseLog: categorical: [True, False]  

## UPCF
Recency: categorical: [1, 5, 25, 100]  
$q$: categorical: [1, 5, 10, 50, 100, 1000]  
$\alpha$: categorical: [0, 0.25, 0.5, 0.75, 1]  
TopkNeighbors: categorical: [None, 10, 100, 300, 600, 900]  
Preprocessing: categorical: [None, "binary"]  

## DNNTSP
NumNearestNeighbors: ctaegorical: [100, 300, 500, 700, 900, 1100, 1300]  
WithinDecayRate: categorical: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
GroupDecayRate: categorical: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
$\alpha$: categorical: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
MaxWindow: int: max_window $\in [3, 10]$



## SAFERec
HiddenDim: categorical: [32, 64, 128, 256]  
NHeads: categorical: [1, 2, 4]   
NLayers: categorical: [1, 2, 4]  
$F_{max}$: int: $F_{max} \in [1, 50]$  
L: categorical: [32, 64, 128, 256]
