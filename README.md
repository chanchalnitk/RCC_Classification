# Data Availability

https://drive.google.com/drive/folders/1Lz7THwUxFQVOlwKx4lfGA0l_dtqC4HkB?usp=sharing


Kidney cancer is a major contributor to cancer-related mortality, with renal cell carcinoma (RCC) being the most common subtype, accounting for approximately 80–85% of all renal tumors. Traditional RCC diagnosis through histopathology image analysis is labor-intensive, error-prone, and heavily reliant on expert interpretation. To overcome these limitations, we propose RenalNet, a robust and computationally efficient deep learning architecture for the accurate classification of RCC subtypes. RenalNet is designed to capture cross-channel and inter-spatial features at three scales simultaneously, enabling the network to learn complex tissue morphology effectively. It incorporates a novel Multiple Channel Residual Transformation (MCRT) module to focus on the most relevant morphological features by fusing information from multiple paths, and a Group Convolutional Deep Localization (GCDL) module that enhances representational power by integrating three distinct feature descriptors. To support this work, we curated a new benchmark dataset from The Cancer Genome Atlas (TCGA), extracting high-quality image patches from hematoxylin and eosin (H\&E) stained whole slide images (WSIs) under expert supervision. RenalNet was rigorously evaluated on three widely used datasets, achieving superior classification accuracies of 91.67%, 97.14%, and 97.24%, outperforming state-of-the-art methods. Additionally, it significantly reduces computational overhead, requiring only 2.71× fewer FLOPs and 0.2131× the parameters, highlighting its effectiveness for real-world clinical applications.









Our key contributions are summarized as follows: (1) We introduce RenalNet, a highly accurate and computationally efficient end-to-end deep learning architecture for classifying RCC subtypes from H\&E-stained kidney histopathology images, developed through a close collaboration between experienced healthcare professionals and AI experts. (2) To enhance the model’s ability to capture fine-grained morphological patterns, we design two novel convolutional modules: Multiple Channel Residual Transformation (MCRT), which emphasizes the most critical tissue features, and Group Convolutional Deep Localization (GCDL), which fuses three complementary feature descriptors to improve localization and discrimination. (3) We also construct a new benchmark dataset derived from whole slide images (WSIs) in The Cancer Genome Atlas (TCGA), with region-of-interest (ROI) extraction guided by expert pathologists, which will be publicly released to support reproducibility and future research. (4) RenalNet demonstrates superior performance across multiple organ-specific histopathology datasets, surpassing eight recent state-of-the-art models while requiring significantly fewer computational resources and training time—making it both scalable and practical for real-world clinical use.

