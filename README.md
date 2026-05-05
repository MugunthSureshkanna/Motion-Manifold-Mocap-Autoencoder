**Motion Manifold — Mocap Autoencoder**

A temporal convolutional autoencoder trained on the CMU Motion Capture dataset to learn a continuous latent manifold of human motion, enabling motion synthesis, style transfer, gap filling, and corruption repair.

**Overview**

This project trains a deep convolutional autoencoder on windowed segments of 3D skeletal motion data. The encoder compresses 160-frame motion windows (joint positions + root velocity) into a compact latent representation; the decoder reconstructs them. Once the manifold is learned, operations like interpolation and style transfer become simple arithmetic in latent space.

**Architecture**

**MotionAutoencoder (AE.py) A symmetric temporal conv-deconv network:**
- Encoder: 3 strided Conv1d blocks (kernel size 15) with channel progression input_dim → 256 → 128 → 64, downsampling by 8×
- Decoder: 3 ConvTranspose1d blocks mirroring the encoder, restoring original sequence length
- Kaiming initialization for encoder weights, Xavier for the final decoder layer
- Optional element-dropout corruption during training to improve manifold robustness

**Input features (dataloader.py)**
- Normalized joint positions (21 joints × 3 dims = 63 features)
- Root translational velocity (XZ plane, 2 features)
- Root rotational velocity (Y axis, 1 feature)
- Total: 66-dimensional feature vector per frame

**Training (MotionManifoldTrainer)**
- Two-phase training: initial (lr=1e-3, corruption=0.1) → fine-tuning (lr=3e-4, no corruption)
- Loss: MSE on joint positions + weighted MSE on root velocity + L1 sparsity on latent activations
- Gradient clipping (max norm 1.0), exponential LR decay, best-checkpoint saving by validation loss
- Trained on TACC (Texas Advanced Computing Center)

**Capabilities**

All operations work by manipulating the learned latent space:

**Motion Interpolation** — Encode two motion clips, linearly interpolate latent vectors at parameter t, decode. Root velocity is interpolated separately and used to recover global trajectory.

**Style Transfer** — Encode content and style clips; apply AdaIN-style latent statistics transfer (standardize content latent, rescale with style mean/std), blend with alpha, decode. Demonstrated: walk→zombie, run→catwalk, punch→wave.

**Gap Filling / Inpainting** — Given a motion with masked frames, replace missing frames with mean pose, encode, decode to fill gaps. Observed and reconstructed clips compared side by side.

**Corruption Repair** — Apply zero-masking, Gaussian noise, or joint dropout to a motion sequence, then project onto the manifold to recover a plausible clean motion.

**Motion Extension** — Encode a partial clip and decode to a longer sequence, extending motion beyond the observed window.

**IMPORTANT**: 
If you want to run the code, please download the .pkl file from the following [link](https://drive.google.com/file/d/1OI1ewtBsp51P37D7cNbhNlDF85zEnd2u/view?usp=sharing) and place it in the cmu-mocap/cache folder.

Afterward, install the required dependencies after creating a python virtual environment by running: pip install -r requirements.txt

Additionally, if you would like to retrain the model on a platform such as TACC, feel free to use run_notebook.sh to run a batch process.
