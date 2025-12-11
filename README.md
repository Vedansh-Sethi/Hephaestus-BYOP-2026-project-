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
        - Latent Space Traveral -
        ![](./assets/latent_space_traversal_beta=2.5_dsprites.png)
        