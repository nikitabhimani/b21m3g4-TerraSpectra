Place the served TorchScript model here as `model.pt` (it is mounted read-only into the api and worker containers at `/models`).

- Before P2's model is ready: `make stub-model` writes the contract stub.
- Afterwards: `make -C model export` writes the trained model.
