import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mlp import (
    TrainingConfig,
    NeuralBetaDataset,
    NeuralBetaMLP,
    NeuralBetaLoss,
    NeuralBetaTrainer,
    DEVICE,
)


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.batch_size == 256
        assert config.lr == 3e-4
        assert config.max_epochs == 50
        assert config.num_workers == 0
        assert config.weight_decay == 1e-4
        assert config.patience == 5

    def test_custom_values(self):
        config = TrainingConfig(batch_size=64, lr=1e-3, max_epochs=100)
        assert config.batch_size == 64
        assert config.lr == 1e-3
        assert config.max_epochs == 100

    def test_device_is_torch_device(self):
        config = TrainingConfig()
        assert isinstance(config.device, torch.device)


class TestNeuralBetaDataset:
    def test_init_valid_data(self):
        X = np.random.randn(100, 5)
        r_next = np.random.randn(100)
        mkt_next = np.random.randn(100)

        dataset = NeuralBetaDataset(X, r_next, mkt_next)

        assert len(dataset) == 100
        assert dataset.X.shape == (100, 5)
        assert dataset.r.shape == (100,)
        assert dataset.m.shape == (100,)

    def test_init_with_lists(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        r_next = [0.1, 0.2, 0.3]
        mkt_next = [0.05, 0.1, 0.15]

        dataset = NeuralBetaDataset(X, r_next, mkt_next)

        assert len(dataset) == 3

    def test_init_length_mismatch_raises(self):
        X = np.random.randn(100, 5)
        r_next = np.random.randn(50)
        mkt_next = np.random.randn(100)

        with pytest.raises(ValueError, match="Length mismatch"):
            NeuralBetaDataset(X, r_next, mkt_next)

    def test_init_mkt_length_mismatch_raises(self):
        X = np.random.randn(100, 5)
        r_next = np.random.randn(100)
        mkt_next = np.random.randn(50)

        with pytest.raises(ValueError, match="Length mismatch"):
            NeuralBetaDataset(X, r_next, mkt_next)

    def test_getitem_returns_dict(self):
        X = np.random.randn(10, 3)
        r_next = np.random.randn(10)
        mkt_next = np.random.randn(10)

        dataset = NeuralBetaDataset(X, r_next, mkt_next)
        item = dataset[0]

        assert isinstance(item, dict)
        assert "x" in item
        assert "r_next" in item
        assert "mkt_next" in item

    def test_getitem_returns_correct_values(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        r_next = np.array([0.1, 0.2])
        mkt_next = np.array([0.05, 0.1])

        dataset = NeuralBetaDataset(X, r_next, mkt_next)
        item = dataset[0]

        assert torch.allclose(item["x"], torch.tensor([1.0, 2.0]))
        assert item["r_next"].item() == pytest.approx(0.1)
        assert item["mkt_next"].item() == pytest.approx(0.05)

    def test_tensors_are_float32(self):
        X = np.random.randn(10, 3)
        r_next = np.random.randn(10)
        mkt_next = np.random.randn(10)

        dataset = NeuralBetaDataset(X, r_next, mkt_next)

        assert dataset.X.dtype == torch.float32
        assert dataset.r.dtype == torch.float32
        assert dataset.m.dtype == torch.float32


class TestNeuralBetaMLP:
    def test_init_default_activation(self):
        model = NeuralBetaMLP(in_dim=5)

        assert isinstance(model.net[1], nn.ReLU)

    def test_init_relu_activation(self):
        model = NeuralBetaMLP(in_dim=5, activation="relu")

        assert isinstance(model.net[1], nn.ReLU)

    def test_init_gelu_activation(self):
        model = NeuralBetaMLP(in_dim=5, activation="gelu")

        assert isinstance(model.net[1], nn.GELU)

    def test_init_tanh_activation(self):
        model = NeuralBetaMLP(in_dim=5, activation="tanh")

        assert isinstance(model.net[1], nn.Tanh)

    def test_init_elu_activation(self):
        model = NeuralBetaMLP(in_dim=5, activation="elu")

        assert isinstance(model.net[1], nn.ELU)

    def test_init_identity_activation(self):
        model = NeuralBetaMLP(in_dim=5, activation="identity")

        assert isinstance(model.net[1], nn.Identity)

    def test_init_linear_activation(self):
        model = NeuralBetaMLP(in_dim=5, activation="linear")

        assert isinstance(model.net[1], nn.Identity)

    def test_init_sigmoid_activation(self):
        model = NeuralBetaMLP(in_dim=5, activation="sigmoid")

        assert isinstance(model.net[1], nn.Sigmoid)

    def test_init_unknown_activation_defaults_to_relu(self):
        model = NeuralBetaMLP(in_dim=5, activation="unknown_act")

        assert isinstance(model.net[1], nn.ReLU)

    def test_init_custom_hidden_dim(self):
        model = NeuralBetaMLP(in_dim=5, hidden_dim=16)

        assert model.net[0].out_features == 16
        assert model.net[2].in_features == 16

    def test_forward_output_shape(self):
        model = NeuralBetaMLP(in_dim=5, hidden_dim=8)
        x = torch.randn(32, 5)

        output = model(x)

        assert output.shape == (32,)

    def test_forward_single_sample(self):
        model = NeuralBetaMLP(in_dim=3)
        x = torch.randn(1, 3)

        output = model(x)

        assert output.shape == (1,)

    def test_forward_deterministic(self):
        torch.manual_seed(42)
        model = NeuralBetaMLP(in_dim=5)
        model.eval()
        x = torch.randn(10, 5)

        output1 = model(x)
        output2 = model(x)

        assert torch.allclose(output1, output2)


class TestNeuralBetaLoss:
    def test_init_default_eps(self):
        loss_fn = NeuralBetaLoss()

        assert loss_fn.eps == 1e-12

    def test_init_custom_eps(self):
        loss_fn = NeuralBetaLoss(eps=1e-8)

        assert loss_fn.eps == 1e-8

    def test_forward_perfect_prediction(self):
        loss_fn = NeuralBetaLoss()

        beta_hat = torch.tensor([1.0, 1.0, 1.0])
        mkt_next = torch.tensor([0.1, 0.2, 0.3])
        r_next = torch.tensor([0.1, 0.2, 0.3])

        loss = loss_fn(beta_hat, mkt_next, r_next)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_forward_nonzero_loss(self):
        loss_fn = NeuralBetaLoss()

        beta_hat = torch.tensor([1.0, 1.0])
        mkt_next = torch.tensor([0.1, 0.2])
        r_next = torch.tensor([0.2, 0.4])

        loss = loss_fn(beta_hat, mkt_next, r_next)

        assert loss.item() > 0

    def test_forward_returns_scalar(self):
        loss_fn = NeuralBetaLoss()

        beta_hat = torch.randn(100)
        mkt_next = torch.randn(100)
        r_next = torch.randn(100)

        loss = loss_fn(beta_hat, mkt_next, r_next)

        assert loss.dim() == 0

    def test_forward_is_rmse(self):
        loss_fn = NeuralBetaLoss(eps=0)

        beta_hat = torch.tensor([2.0, 2.0])
        mkt_next = torch.tensor([1.0, 1.0])
        r_next = torch.tensor([3.0, 3.0])

        loss = loss_fn(beta_hat, mkt_next, r_next)
        expected = torch.sqrt(torch.tensor(1.0))

        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_forward_gradient_flows(self):
        loss_fn = NeuralBetaLoss()

        beta_hat = torch.tensor([1.0, 1.0], requires_grad=True)
        mkt_next = torch.tensor([0.1, 0.2])
        r_next = torch.tensor([0.2, 0.4])

        loss = loss_fn(beta_hat, mkt_next, r_next)
        loss.backward()

        assert beta_hat.grad is not None


class TestNeuralBetaTrainer:
    @pytest.fixture
    def simple_model(self):
        return NeuralBetaMLP(in_dim=5, hidden_dim=4)

    @pytest.fixture
    def simple_dataset(self):
        X = np.random.randn(100, 5)
        r_next = np.random.randn(100)
        mkt_next = np.random.randn(100)
        return NeuralBetaDataset(X, r_next, mkt_next)

    def test_init(self, simple_model):
        trainer = NeuralBetaTrainer(simple_model)

        assert trainer.model is simple_model
        assert isinstance(trainer.config, TrainingConfig)
        assert isinstance(trainer.loss_fn, NeuralBetaLoss)
        assert trainer.best_val_loss == float("inf")

    def test_init_model_on_device(self, simple_model):
        trainer = NeuralBetaTrainer(simple_model)

        param = next(trainer.model.parameters())
        assert param.device.type == trainer.device.type

    def test_train_one_epoch_returns_float(self, simple_model, simple_dataset):
        trainer = NeuralBetaTrainer(simple_model)
        loader = DataLoader(simple_dataset, batch_size=16)
        trainer.optimizer = torch.optim.Adam(simple_model.parameters())

        loss = trainer.train_one_epoch(loader)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate_returns_float(self, simple_model, simple_dataset):
        trainer = NeuralBetaTrainer(simple_model)
        loader = DataLoader(simple_dataset, batch_size=16)

        loss = trainer.evaluate(loader)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate_no_grad(self, simple_model, simple_dataset):
        trainer = NeuralBetaTrainer(simple_model)
        loader = DataLoader(simple_dataset, batch_size=16)

        trainer.evaluate(loader)

        for param in simple_model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_fit_returns_dict(self, simple_model, simple_dataset):
        trainer = NeuralBetaTrainer(simple_model)
        trainer.config.max_epochs = 2

        result = trainer.fit(simple_dataset, simple_dataset)

        assert isinstance(result, dict)
        assert "model" in result
        assert "history" in result
        assert "best_val" in result

    def test_fit_history_contains_losses(self, simple_model, simple_dataset):
        trainer = NeuralBetaTrainer(simple_model)
        trainer.config.max_epochs = 3

        result = trainer.fit(simple_dataset, simple_dataset)

        assert len(result["history"]["train"]) > 0
        assert len(result["history"]["val"]) > 0
        assert len(result["history"]["lr"]) > 0

    def test_fit_early_stopping(self, simple_model, simple_dataset):
        trainer = NeuralBetaTrainer(simple_model)
        trainer.config.max_epochs = 100
        trainer.config.patience = 2

        result = trainer.fit(simple_dataset, simple_dataset)

        assert len(result["history"]["train"]) <= 100

    def test_fit_loads_best_state(self, simple_model, simple_dataset):
        trainer = NeuralBetaTrainer(simple_model)
        trainer.config.max_epochs = 5

        result = trainer.fit(simple_dataset, simple_dataset)

        assert result["model"] is simple_model


class TestDataLoaderIntegration:
    def test_dataset_works_with_dataloader(self):
        X = np.random.randn(100, 5)
        r_next = np.random.randn(100)
        mkt_next = np.random.randn(100)
        dataset = NeuralBetaDataset(X, r_next, mkt_next)

        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        batch = next(iter(loader))
        assert batch["x"].shape == (16, 5)
        assert batch["r_next"].shape == (16,)
        assert batch["mkt_next"].shape == (16,)

    def test_full_training_loop(self):
        X = np.random.randn(50, 3)
        r_next = np.random.randn(50)
        mkt_next = np.random.randn(50)

        train_ds = NeuralBetaDataset(X[:40], r_next[:40], mkt_next[:40])
        val_ds = NeuralBetaDataset(X[40:], r_next[40:], mkt_next[40:])

        model = NeuralBetaMLP(in_dim=3, hidden_dim=4)
        trainer = NeuralBetaTrainer(model)
        trainer.config.max_epochs = 3
        trainer.config.batch_size = 8

        result = trainer.fit(train_ds, val_ds)

        assert result["best_val"] < float("inf")
        assert len(result["history"]["train"]) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
