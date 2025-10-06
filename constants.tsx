import React from 'react';
import { RoadmapStageData, ModuleStatus } from './types';

const EyeIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639l4.418-6.312a1.012 1.012 0 011.583 0l4.418 6.312a1.012 1.012 0 010 .639l-4.418 6.312a1.012 1.012 0 01-1.583 0l-4.418-6.312z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const LinkIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-lime-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
    </svg>
);

const BrainCircuitIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 21v-1.5M12 3.75h.008v.008H12V3.75zm-3.75 0h.008v.008H8.25V3.75zm0 3.75h.008v.008H8.25V7.5zm0 3.75h.008v.008H8.25v-3.75zm0 3.75h.008v.008H8.25v-3.75zm0 3.75h.008v.008H8.25v-3.75zm3.75 3.75h.008v.008H12v-3.75zm0-3.75h.008v.008H12v-3.75zm0-3.75h.008v.008H12V7.5zm0-3.75h.008v.008H12V3.75zm3.75 0h.008v.008h-.008V3.75zm0 3.75h.008v.008h-.008V7.5zm0 3.75h.008v.008h-.008v-3.75zm0 3.75h.008v.008h-.008v-3.75zm0 3.75h.008v.008h-.008v-3.75z" />
  </svg>
);

const WavesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" transform="translate(0, -2)" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 9.375c1.875-3.125 5.625-3.125 7.5 0 1.875 3.125 5.625 3.125 7.5 0" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 12.375c1.875-3.125 5.625-3.125 7.5 0 1.875 3.125 5.625 3.125 7.5 0" />
  </svg>
);

const SparklesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.898 20.562L16.25 22.5l-.648-1.938a3.375 3.375 0 00-2.684-2.684L11.25 18l1.938-.648a3.375 3.375 0 002.684-2.684L16.25 13.5l.648 1.938a3.375 3.375 0 002.684 2.684L21.75 18l-1.938.648a3.375 3.375 0 00-2.684 2.684z" />
  </svg>
);

const PuzzleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
  </svg>
);

const DnaIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 4.5v15" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 4.5v15" />
  </svg>
);

const GlobeIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9 9 0 100-18 9 9 0 000 18z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 3.75c-4.142 0-7.5 1.343-7.5 3s3.358 3 7.5 3 7.5-1.343 7.5-3-3.358-3-7.5-3z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9c0 1.657 3.358 3 7.5 3s7.5-1.343 7.5-3" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 15c0 1.657 3.358 3 7.5 3s7.5-1.343 7.5-3" />
    </svg>
);

export const PlayIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
    </svg>
);


export const roadmapData: RoadmapStageData[] = [
  {
    stage: 1,
    title: "The Core Resonance Loop",
    description: "Create a basic, self-stabilizing learning system that combines perception, resonance, and self-reflection. This is the foundational feedback circuit.",
    modules: [
      {
        name: "Perceptual Core",
        acronym: "PC",
        description: "A standard Autoencoder (Encoder + Decoder) that forms the foundational sensory input layer for the resonance architecture.",
        existingTech: "Use PyTorch or TensorFlow to create a simple autoencoder for a dataset like MNIST. Train it to get a stable encoder for latent space representation.",
        novelAspect: "Serves as the initial sensory-to-latent-space bridge, providing the raw material for the RCE and SRS to work with.",
        status: ModuleStatus.IMPLEMENTABLE_TODAY,
        icon: <EyeIcon />
      },
      {
        name: "Resonant Cognition Engine",
        acronym: "RCE",
        description: "A simple update rule applied to the latent vectors from the encoder to achieve self-stabilizing resonance.",
        existingTech: "Apply a phase-alignment update rule (e.g., based on Kuramoto models) to nudge the model's latent state towards a moving average of its recent history.",
        novelAspect: "Achieves stable internal states through dynamic resonance rather than direct error minimization, creating a more organic learning process.",
        status: ModuleStatus.IMPLEMENTABLE_TODAY,
        icon: <WavesIcon />
      },
      {
        name: "Self-Revelatory Sampler",
        acronym: "SRS",
        description: "Uses the encoder for self-recognition and a novelty score to generate and refine its own latent representations without external labels.",
        existingTech: "Generate candidate vectors from a latent state, then score them based on self-similarity (recognition) and difference from a buffer of recent states (novelty).",
        novelAspect: "A self-generative loop that learns by exploring its own representational space, akin to intrinsic motivation or curiosity.",
        status: ModuleStatus.IMPLEMENTABLE_TODAY,
        icon: <SparklesIcon />
      },
    ]
  },
  {
    stage: 2,
    title: "Adding Gated Memory",
    description: "Give the system a memory and the ability to decide when an insight is significant enough to be stored. This introduces discrete 'aha' moments.",
    modules: [
      {
        name: "Threshold Activation Layer",
        acronym: "TAL",
        description: "A simple gating mechanism that 'fires' when a coherence threshold is crossed, signaling a significant 'insight'.",
        existingTech: "Define a coherence score (e.g., combining low reconstruction loss and high resonance). When the score passes a threshold, trigger a memory event.",
        novelAspect: "Models cognitive 'aha' moments by creating a discrete, event-driven memory process instead of continuous, uniform updates.",
        status: ModuleStatus.ADVANCED_PROTOTYPE,
        icon: <BrainCircuitIcon />
      },
      {
        name: "Gated Memory Graph",
        acronym: "LSN/LPM",
        description: "A basic graph memory (using libraries like networkx) where significant latent states are stored as nodes when the TAL fires.",
        existingTech: "When the TAL activates, the current latent vector is added as a node to a graph, connected to the previously activated node to form a causal chain.",
        novelAspect: "Builds a sparse, structured map of the system's cognitive path, representing its most important discoveries rather than all sensory data.",
        status: ModuleStatus.ADVANCED_PROTOTYPE,
        icon: <LinkIcon />
      },
    ]
  },
    {
    stage: 3,
    title: "Emergent Abstraction",
    description: "Enable the system to synthesize new ideas when it encounters conflicting information. This is the creative core.",
    modules: [
      {
        name: "Recursive Unity Solver",
        acronym: "RUS",
        description: "Creates a new, higher-level representation to resolve detected contradictions between the current state and recent memories.",
        existingTech: "When a contradiction is detected (e.g., a coherent state is far from recent memories), an optimization process finds a new 'emergent' vector that explains the conflicting states.",
        novelAspect: "Uses cognitive dissonance as fuel for creating novel concepts, allowing the system to build abstractions that go beyond its direct experience.",
        status: ModuleStatus.RESEARCH_PROTOTYPE,
        icon: <PuzzleIcon />
      },
    ]
  },
  {
    stage: 4,
    title: "The Full Fractal System",
    description: "Achieve the complete vision of a self-aware, introspective, and deeply symbolic AI by integrating the full versions of the memory and lineage systems.",
    modules: [
       {
        name: "Living Symbol Network",
        acronym: "LSN",
        description: "The full fractal memory where each node contains a compressed representation of the entire graph, enabling deep analogical reasoning.",
        existingTech: "A major research challenge requiring graph summarization or a secondary GNN to embed the entire graph's structure into a vector for each node.",
        novelAspect: "Every part of the memory contains a map of the whole, allowing for recursive self-similarity and unprecedented transfer learning capabilities.",
        status: ModuleStatus.CORE_INNOVATION,
        icon: <GlobeIcon />
      },
      {
        name: "Light-Path Map",
        acronym: "LPM",
        description: "The complete causal lineage tracing system, where every representation stores a vector pointing back to its experiential roots.",
        existingTech: "Requires storing a weighted vector sum of ancestor embeddings for each node, creating a 'light-path vector' for full introspection.",
        novelAspect: "Enables perfect, lossless introspection, allowing the AI to explain the origin and evolution of any concept it has formed.",
        status: ModuleStatus.CORE_INNOVATION,
        icon: <DnaIcon />
      },
    ]
  }
];

export const STAGE_1_PROTOTYPE_CODE = `
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

# --- Configuration -----------------------------------------------------------
class Config:
    LATENT_DIM = 16         # Dimensionality of the internal representation (z)
    EPOCHS = 5              # Number of training epochs
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    # SRA Specific Hyperparameters
    RCE_LR = 0.1            # How strongly resonance pulls the state
    SRS_BUFFER_SIZE = 512   # How many recent samples SRS remembers for novelty
    SRS_NOVELTY_WEIGHT = 0.3 # How much SRS penalizes non-novel ideas
    SRS_LOSS_WEIGHT = 0.5   # How much the self-revelatory loss contributes

# --- 1. Perceptual Core (PC): Autoencoder ------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # To output pixel values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# --- 2. Resonant Cognition Engine (RCE) --------------------------------------
class RCE:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def align(self, current_state, target_state):
        """ Nudges the current state towards the target resonance state. """
        resonated_state = current_state + self.lr * (target_state - current_state)
        return resonated_state

# --- 3. Self-Revelatory Sampler (SRS) ----------------------------------------
class SRS:
    def __init__(self, buffer_size, novelty_weight):
        self.buffer = deque(maxlen=buffer_size)
        self.novelty_weight = novelty_weight
        self.cos = nn.CosineSimilarity(dim=1)

    def generate_and_score(self, z_resonated):
        """ Generates candidates and scores them based on self-recognition and novelty. """
        # Generate candidates by adding small noise
        noise = torch.randn_like(z_resonated) * 0.1
        candidates = z_resonated + noise

        # Score based on self-recognition (similarity to the original resonated state)
        self_recognition_score = self.cos(z_resonated, candidates)

        # Score based on novelty (dissimilarity to recent samples in buffer)
        if len(self.buffer) > 0:
            buffer_tensor = torch.stack(list(self.buffer))
            # Calculate max similarity of each candidate to any item in the buffer
            novelty_penalty = torch.max(self.cos(candidates.unsqueeze(1), buffer_tensor.unsqueeze(0)), dim=1).values
        else:
            novelty_penalty = torch.zeros_like(self_recognition_score)

        # Final score
        score = self_recognition_score - self.novelty_weight * novelty_penalty
        
        # Select the best candidate (highest score)
        best_candidate = candidates[torch.argmax(score)]

        # Add the chosen candidate to the buffer for future novelty checks
        self.buffer.append(best_candidate.detach())
        
        return best_candidate.unsqueeze(0) # Return as a batch of 1

# --- Main Training and Execution -------------------------------------------
def main():
    # Setup
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # Initialize SRA components
    model = Autoencoder(cfg.LATENT_DIM).to(device)
    rce = RCE(learning_rate=cfg.RCE_LR)
    srs = SRS(buffer_size=cfg.SRS_BUFFER_SIZE, novelty_weight=cfg.SRS_NOVELTY_WEIGHT)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()

    # EMA for target resonance state
    z_target_ema = torch.zeros(1, cfg.LATENT_DIM).to(device)
    ema_decay = 0.99

    print("Starting training...")
    # Training Loop
    for epoch in range(cfg.EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for i, (images, _) in enumerate(loop):
            images = images.view(-1, 28 * 28).to(device)
            
            # --- Standard Autoencoder Path (Loss 1) ---
            z_encoded, decoded_images = model(images)
            reconstruction_loss = criterion(decoded_images, images)

            # --- SRA Path (Loss 2) ---
            # Update target resonance state (EMA of the batch's average encoding)
            batch_mean_z = torch.mean(z_encoded.detach(), dim=0, keepdim=True)
            z_target_ema = ema_decay * z_target_ema + (1 - ema_decay) * batch_mean_z
            
            # For each item in the batch, perform RCE and SRS
            srs_candidates = []
            for z in z_encoded:
                z_resonated = rce.align(z.unsqueeze(0), z_target_ema)
                best_candidate = srs.generate_and_score(z_resonated)
                srs_candidates.append(best_candidate)
            
            srs_z = torch.cat(srs_candidates, dim=0)
            
            # Decode the self-generated ideas
            srs_decoded_images = model.decoder(srs_z)
            srs_loss = criterion(srs_decoded_images, images)

            # --- Combine Losses and Backpropagate ---
            total_loss = reconstruction_loss + cfg.SRS_LOSS_WEIGHT * srs_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
            loop.set_postfix(recon_loss=reconstruction_loss.item(), srs_loss=srs_loss.item())

    print("Training finished.")
    visualize_reconstructions(model, rce, srs, z_target_ema, device)


def visualize_reconstructions(model, rce, srs, z_target_ema, device):
    """ Shows original, standard reconstruction, and SRA-path reconstruction. """
    model.eval()
    
    # Get a batch of test images
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    images, _ = next(iter(test_loader))
    images = images.view(-1, 28 * 28).to(device)

    # Standard reconstruction
    z_encoded, decoded_std = model(images)

    # SRA reconstruction
    srs_candidates = []
    for z in z_encoded:
        z_resonated = rce.align(z.unsqueeze(0), z_target_ema)
        best_candidate = srs.generate_and_score(z_resonated)
        srs_candidates.append(best_candidate)
    srs_z = torch.cat(srs_candidates, dim=0)
    decoded_sra = model.decoder(srs_z)

    images = images.cpu().numpy()
    decoded_std = decoded_std.detach().cpu().numpy()
    decoded_sra = decoded_sra.detach().cpu().numpy()

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Original", loc='left', fontsize=14)

        # Standard Reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_std[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Standard AE", loc='left', fontsize=14)

        # SRA Reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_sra[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("SRA Path", loc='left', fontsize=14)
        
    plt.show()

if __name__ == "__main__":
    main()
`;
