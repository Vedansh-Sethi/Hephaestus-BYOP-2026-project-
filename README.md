# BYOP 2026 project

## Aim of Project -

- The project is to be implemented in 3 stages
- The first stage :- It aims to identify outliers using context of a batch of input pictures
- The second stage :- It aims to interpret the reason why the model ranked the pictures the way it did
- The third stage :- Using generative models, I plan to explore the latent space with the context of input pictures

---

## Stage 1 : Experimentation -

- Training of VAE on MNIST dataset -

  - Model Architecture -

    - Encoder - 784 => 400 => 20 (mu, logvar) (all layers use ReLU activation)
    - Decoder - 20 (mu + eps \* std) => 400 => 784 (all layers use ReLU activation except last one which uses sigmoid)
    - Optimiser - Adam
    - LR = 1e-3
  - Results -

    - Loss graph -
    ![](./assets/loss_graph_MNIST.png)
    - Reconstructued Images -
    ![](./assets/reconstruction_MNIST.png)
    - Morphing results -
    ![](./assets/morphing9to4.png)
    ![](./assets/morphing3to8.png)
    ![](./assets/morphing1to7.png)

- Training Beta VAE on dsprites dataset -

  - Model Architecure -

    - Encoder - 4096 => 1024 => 256 => 64 => 10 (mu, logvar) (all layers have ReLU activation except last)
    - Decoder - 10 (mu + eps \* std) => 64 => 256 => 1024 => 4096
      (all layers have ReLU activation except last, which has sigmoid activation)
    - Optimiser - Adam
    - LR = 1e-3

  - Results for different value of Beta -

    - Beta = 1

      - Loss graph -
      ![](./assets/loss_graph_beta=1_dsprites.png)
      - Reconstruction -
      ![](./assets/reconstruction_beta=1_dsprites.png)
      - Latent Space Traversal -
      ![](./assets/latent_space_traversal_beta=1_dsprites.png)
    

    - Beta = 4

        - Loss graph -
        ![](./assets/loss_graph_beta=4_dsprites.png)
        - Reconstruction -
        ![](./assets/reconstruction_beta=4_dsprites.png)
        - Latent Space Traversal -
        ![](./assets/latent_space_traversal_beta=4_dsprites.png)
    
    - Beta = 2.5 

        - Loss graph -
        ![](./assets/loss_graph_beta=2.5_dsprites.png)
        - Reconstruction -
        ![](./assets/reconstruction_beta=2.5_dsprites.png)
        - Latent Space Traversal -
        ![](./assets/latent_space_traversal_beta=2.5_dsprites.png)
      

- Training Gamma VAE on dsprites -

  - Model Architecture (Feed Forward Network) -

    - Encoder - 4096 => 1024 => 256 => 64 => 10 (mu, logvar) (All layers have ReLU activation except last one)
    - Decoder - 10 => 64 => 256 => 1024 => 4096 
    (all layers have ReLU activation except last, which has sigmoid activation)
    - Optimiser - Adam
  
  - Hyperparameters (1) -

    - LR = 1e-3
    - Epochs - 80
    - Cutoff epoch - 60
    - Gamma - 100
    - Maximum capacity (per image) - 25
  
  - Results -
    
    - Loss graph -
      ![](./assets/loss_graph_GammaVAE_linear.png)
    - Reconstruction -
      ![](./assets/reconstruction_GammaVAE_linear.png)
    - Latent Space Traversal -
      ![](./assets/latent_traversal_GammaVAE_linear.png)
  
  - Hyperparameters (2) -

    - LR = 5e-4
    - Epochs - 80
    - Cutoff epoch - 60
    - Gamma Max - 100
    - Gamma Min - 20
    - Maximum capacity (per image) - 25

  - Results -
    
    - Loss graph -
      ![](./assets/loss_graph_decayingGammaVAE_linear.png)
    - Reconstruction -
      ![](./assets/reconstruction_decayingGammaVAE_linear.png)
    - Latent Space Traversal -
      ![](./assets/latent_traversal_decayingGammaVAE_linear.png)
    
  - Model Architecture (CNNs) -

    - Encoder - (1, 64, 64) => (32, 32, 32) => (32, 16, 16) => (32, 8, 8) => (32, 4, 4) -> 512 => 256 => 64 => 10 (mu, logvar)
    - Decoder - 10 => 64 => 256 => 512 -> (32, 4, 4) => (32, 8, 8) => (32, 16, 16) => (32, 32, 32) => (1, 64, 64)
    - Optimiser - Adam
  
  - Hyperparameters (1) -

    - LR = 5e-4
    - Epochs - 80
    - Cutoff Epoch - 60
    - Gamma - 10
    - Maximum Capacity (per image) - 25

  - Results -

    - Loss Graph -
      ![](./assets/loss_graph_GammaVAE_CNN.png)
    - Reconstruction -
      ![](./assets/reconstruction_GammaVAE_CNN.png)
    - Latent Space Traversal -
      ![](./assets/latent_traversal_GammaVAE_CNN.png)
  
  - Hyperparameters (2) -
    
    - LR = 5e-4
    - Epochs - 100
    - Cutoff epochs - 60
    - Gamma max - 10
    - Gamma min - 2
    - Maximum Capacity (per image) - 25
  
  - Results - 
    
    - Loss graph -
      ![](./assets/loss_graph_decayingGammaVAE_CNN.png)
    - Reconstruction -
      ![](./assets/reconstruction_decayingGammaVAE_CNN.png)
    - Latent Space Traversal -
      ![](./assets/latent_traversal_decayingGammaVAE_CNN.png)

  - Training Beta TCVAE on dsprites -
    
    - Model Architecture -

      - Encoder - (1, 64, 64) => (32, 32, 32) => (32, 16, 16) => (32, 8, 8) => (32, 4, 4) ->(32 * 4 * 4) => 256 => 10 (mu, logvar)
      - Decoder - 10 => 256 => 32 * 4 * 4 -> (32, 4, 4) => (32, 8, 8) => (32, 16, 16) => (32, 32, 32) => (1, 64, 64)

    - Hyperparameters -

      - Batch Size = 256
      - Training Batches = 500
      - LR = 5e-4
      - Epochs = 120
      - Gamma = 1
      - Beta = 4.5
      - Anneal Steps = 5000
    
    - Results -

      - Loss Graph -
      ![](./assets/ELBO_loss_BetaTCVAE.png)
      ![](./assets/loss_graph_BetaTCVAE.png)
      - Reconstruction -
      ![](./assets/reconstruction_BetaTCVAE.png)
      - Latent Traversal -
      ![](./assets/latent_traversal_BetaTCVAE.png)

---

## Stage 2 : Training of Outlier Detection Model -

  - Training Beta TCVAE on Shapes3d dataset -

    - Model Architecture -

      - Encoder - (3, 64, 64) => (32, 32, 32) => (64, 16, 16) => (128, 8, 8) => (256, 4, 4)-> 4096 => 256 => 12 (mu, logvar)
      - Decoder - 12 => 256 => 4096 -> (256, 4, 4) => (128, 8, 8) => (64, 16, 16) => (32, 32, 32) => (3, 64, 64)

    - Hyparameters -

      - Batch Size = 128
      - Training = 500
      - LR = 5e-4
      - Epochs = 50
      - Gamma = 1
      - Beta = 6
      - Anneal Steps = 5000
      - Latent Dimensions = 12
    
    - Results -

      - Loss Graph -
      ![](./assets/loss_graph_BetaTCVAE_shapes3d.png)
      ![](./assets/loss_graph_ELBO_BetaTCVAE_shapes3d.png)
      - Reconstruction -
      ![](./assets/reconstruction_BetaTCVAE_shapes3d.png)
      - Latent Traversal -
      ![](./assets/latent_traversal_BetaTCVAE_shapes3d.png)

      - SSIM score = 0.9195
      - MIG score = 0.6235

      - Correlation Matrix -
      ![](./assets/correlation_matrix_BetaTCVAE_shapes3d.png)
    
  - Outlier Detection Results -

    - Semantic outlier detection -
      - 350 samples, 25 outliers and 325 normal samples
      - Normal criteria - fixed shape, wall hue and object hue, variable floor hue, scale and orientation
      - outlier criteria - any object other than the ones in normal criteria

      - Results -
        - Line Graph of MD score -
        ![](./assets/outlier_detection_MD_scores.png)
        The outliers are at the end of the batch, as we can see, their MD scores are significantly higher than normal images
        - Histogram of MD score density -
        ![](./assets/outlier_detection_histogram.png)
        The blue bars are density of MD scores of normal images, whereas the red ones are for outlier images, as we can see, there is nearly perfect seaparation between the two
        - Area Under ROC score - 0.9990