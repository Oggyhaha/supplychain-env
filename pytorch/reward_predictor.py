# pytorch/reward_predictor.py
# ═══════════════════════════════════════════════
# PYTORCH REWARD PREDICTOR
#
# Neural network that predicts reward score
# from warehouse state features.
#
# Architecture:
#   Input:  warehouse state features (numeric)
#   Hidden: 2 layers with ReLU activation
#   Output: predicted reward (0.0 to 1.0)
# ═══════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple
from env.models import WarehouseObservation, SupplyChainReward


class RewardPredictor(nn.Module):
    """
    Neural network that predicts reward from state.

    INPUT FEATURES (what we feed in):
    For each SKU:
      - inventory ratio (current/max_capacity)
      - days of stock remaining
      - demand forecast average
      - reorder urgency (0 or 1)

    Plus global features:
      - budget ratio (remaining/total)
      - day progress (current_day/total_days)
      - pending orders count
      - number of active SKUs

    OUTPUT:
      - predicted reward score (0.0 to 1.0)
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
    ):
        """
        input_size:  number of input features
        hidden_size: neurons in hidden layers
        """
        super(RewardPredictor, self).__init__()

        # Neural network layers
        self.network = nn.Sequential(
            # Layer 1: input → hidden
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),          # Activation function
            nn.Dropout(0.2),    # Prevents overfitting

            # Layer 2: hidden → hidden
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3: hidden → output
            nn.Linear(hidden_size // 2, 16),
            nn.ReLU(),

            # Output layer
            nn.Linear(16, 1),
            nn.Sigmoid(),       # Squeezes output to 0-1
        )

        # Track training history
        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        x: input tensor of shape (batch_size, input_size)
        returns: predicted rewards of shape (batch_size, 1)
        """
        return self.network(x)

    def predict(self, features: np.ndarray) -> float:
        """
        Predict reward for a single state.
        Returns float between 0.0 and 1.0
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            prediction = self.forward(x)
            return float(prediction.item())


def extract_features(
    observation: WarehouseObservation,
    max_skus: int = 4,
) -> np.ndarray:
    """
    Converts warehouse observation into
    a fixed-size numeric feature vector.

    Neural networks need numbers, not text.
    This function does the conversion.

    Returns numpy array of shape (input_size,)
    """
    features = []

    # ── GLOBAL FEATURES ────────────────────────

    # Budget ratio: how much budget is left?
    # 1.0 = full budget, 0.0 = no budget left
    budget_ratio = (
        observation.budget_remaining / observation.total_budget
        if observation.total_budget > 0 else 0.0
    )
    features.append(budget_ratio)

    # Day progress: how far through episode?
    # 0.0 = day 1, 1.0 = last day
    day_progress = (
        observation.current_day / observation.total_days
        if observation.total_days > 0 else 0.0
    )
    features.append(day_progress)

    # Pending orders count (normalized)
    pending_count = min(len(observation.pending_orders) / 10, 1.0)
    features.append(pending_count)

    # Days remaining (normalized)
    days_remaining = (
        observation.days_remaining / observation.total_days
        if observation.total_days > 0 else 0.0
    )
    features.append(days_remaining)

    # ── PER-SKU FEATURES ───────────────────────
    # We extract features for up to max_skus SKUs
    # Pad with zeros if fewer SKUs

    sku_count = 0
    for sku in observation.skus[:max_skus]:
        sku_id      = sku.sku_id
        current_inv = observation.inventory.get(sku_id, 0)
        forecast    = observation.demand_forecast.get(
            sku_id, [10.0] * 7
        )
        avg_demand  = sum(forecast) / len(forecast)

        # Inventory ratio: how full is stock?
        inv_ratio = min(current_inv / sku.max_capacity, 1.0)
        features.append(inv_ratio)

        # Days of stock left
        days_of_stock = (
            min(current_inv / avg_demand, 30.0) / 30.0
            if avg_demand > 0 else 1.0
        )
        features.append(days_of_stock)

        # Is reorder needed? (below reorder point)
        needs_reorder = (
            1.0 if current_inv <= sku.reorder_point else 0.0
        )
        features.append(needs_reorder)

        # Demand forecast normalized
        demand_norm = min(avg_demand / 100.0, 1.0)
        features.append(demand_norm)

        sku_count += 1

    # Pad with zeros for missing SKUs
    for _ in range(max_skus - sku_count):
        features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


class RewardPredictorTrainer:
    """
    Trains the RewardPredictor using experience
    collected from environment interactions.

    Training process:
    1. Collect (state, actual_reward) pairs
    2. Extract features from states
    3. Train network to predict rewards
    4. Validate on held-out data
    """

    def __init__(
        self,
        model: RewardPredictor,
        learning_rate: float = 0.001,
    ):
        self.model     = model
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        self.criterion = nn.MSELoss()  # Mean squared error loss

        # Experience buffer
        self.states:  List[np.ndarray] = []
        self.rewards: List[float]      = []

    def add_experience(
        self,
        observation: WarehouseObservation,
        reward: SupplyChainReward,
    ):
        """
        Adds one (state, reward) pair to buffer.
        Called after each environment step.
        """
        features = extract_features(observation)
        self.states.append(features)
        self.rewards.append(reward.total_score)

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Trains the model on collected experience.

        Returns training history (losses per epoch)
        """
        if len(self.states) < batch_size:
            print(
                f"Not enough data to train. "
                f"Need {batch_size}, have {len(self.states)}"
            )
            return {"train_loss": [], "val_loss": []}

        # Convert to tensors
        X = torch.FloatTensor(np.array(self.states))
        y = torch.FloatTensor(self.rewards).unsqueeze(1)

        # Split into train/validation (80/20)
        split     = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_losses = []
        val_losses   = []

        self.model.train()

        for epoch in range(epochs):

            # Mini-batch training
            permutation = torch.randperm(len(X_train))
            epoch_loss  = 0.0
            batches     = 0

            for i in range(0, len(X_train), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]

                # Forward pass
                predictions = self.model(batch_X)
                loss        = self.criterion(predictions, batch_y)

                # Backward pass (learning happens here)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batches    += 1

            avg_train_loss = epoch_loss / max(batches, 1)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.criterion(
                    val_predictions, y_val
                ).item()
            self.model.train()

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs}: "
                    f"train_loss={avg_train_loss:.4f} "
                    f"val_loss={val_loss:.4f}"
                )

        self.model.training_losses   = train_losses
        self.model.validation_losses = val_losses

        return {
            "train_loss": train_losses,
            "val_loss":   val_losses,
        }

    def save_model(self, path: str = "pytorch/model.pt"):
        """Saves trained model to disk"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = "pytorch/model.pt"):
        """Loads trained model from disk"""
        self.model.load_state_dict(
            torch.load(path, map_location="cpu")
        )
        print(f"Model loaded from {path}")


def train_reward_predictor():
    """
    Complete training pipeline.

    1. Runs environment to collect data
    2. Trains reward predictor
    3. Saves model
    4. Tests predictions
    """
    print("=" * 50)
    print("TRAINING PYTORCH REWARD PREDICTOR")
    print("=" * 50)

    from env.environment import SupplyChainEnvironment
    from env.models import RestockAction, OrderItem

    # Create model and trainer
    model   = RewardPredictor(input_size=20, hidden_size=64)
    trainer = RewardPredictorTrainer(model, learning_rate=0.001)

    # Collect experience by running environment
    print("\nCollecting training data from environment...")
    env    = SupplyChainEnvironment("task_easy")
    result = env.reset()

    step = 0
    while not result.done and step < 30:
        obs = result.observation

        # Simple rule-based policy to collect data
        orders = []
        for sku in obs.skus:
            current = obs.inventory.get(sku.sku_id, 0)
            if current < sku.reorder_point * 2:
                orders.append(OrderItem(
                    sku_id=sku.sku_id,
                    supplier_id="SUP-001",
                    quantity=sku.reorder_point * 3,
                ))

        action = RestockAction(
            orders=orders,
            reasoning="rule_based_collection"
        )

        result = env.step(action)

        # Add to training buffer
        trainer.add_experience(obs, result.reward)
        step += 1

    print(f"Collected {len(trainer.states)} data points")

    # Train the model
    print("\nTraining neural network...")
    history = trainer.train(epochs=50, batch_size=8)

    # Save model
    trainer.save_model("pytorch/model.pt")

    # Test prediction
    print("\nTesting predictions...")
    env    = SupplyChainEnvironment("task_easy")
    result = env.reset()
    obs    = result.observation

    features  = extract_features(obs)
    predicted = model.predict(features)
    print(f"  Sample predicted reward: {predicted:.4f}")
    print(f"  Feature vector size: {len(features)}")

    print("\nPyTorch reward predictor ready!")
    return model


if __name__ == "__main__":
    train_reward_predictor()