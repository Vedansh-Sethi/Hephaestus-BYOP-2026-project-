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

    - Encoder - (1, 64, 64) => (32, 32, 32) => (32, 16, 16) => (32, 8, 8) => (32, 4, 4) -> 512 => 256 => 64 => 10 (myu, logvar)
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
  