# HALO: Hybrid Attention Model for Subcellular Localization

We propose **HALO (Hybrid Attention Model for Subcellular
LOcalization)**, a novel framework that integrates structural information from the AlphaFold structure database and
semantic embeddings from large-scale protein language models, such as fine-tuned ESM (Evolutionary Scale Modeling).
The hybrid architecture uses graph attention networks (GATs) to incorporate biochemical, structural, and sequence-
derived features into a unified representation. Our model leverages ESM embeddings to capture evolutionary and
contextual knowledge of amino acid sequences, complemented by spatial information from AlphaFold-predicted structures.
In addition, it integrates amino acid biochemical properties, such as polarity and hydrophobicity, to enhance feature
diversity. A learnable weighted mechanism dynamically balances contributions from these distinct feature modalities.
We evaluate HALO on three datasets with minimal homology between the training and test sets, where the model
achieves state-of-the-art performance across key metrics. The hybrid approach effectively predicts subcellular localization
for proteins with and without structural data, bridging the gap in datasets where structural information is unavailable. In
particular, the model incorporates a focal loss function with a learnable threshold to address label imbalance and enhance
generalization. We showcase the potential of combining fine-tuned large-language models and AlphaFold structural
embeddings in graph-based neural networks, setting a new standard in subcellular localization prediction. HALO offers a
flexible, adaptable framework for broader protein function annotation tasks, emphasizing the importance of multi-modal
data integration in bioinformatics.

### Figures

![Architecture](https://github.com/user-attachments/assets/94573baa-bfeb-4a2f-b15b-91407b12b489)  
**Figure 1:** Architecture Diagram

![Fine-Tune-ESM](https://github.com/user-attachments/assets/dfc8c5fc-485d-4a11-8bd7-763d0a192d6a)  
**Figure 2:** Fine-tuning ESM-2 Architecture Diagram

![GAT-Model](https://github.com/user-attachments/assets/154a4015-432c-428f-ae0e-25ae9a676e60)  
**Figure 3:** Graph Attention Model Training Diagram
