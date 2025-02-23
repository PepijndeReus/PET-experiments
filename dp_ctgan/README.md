## TODO

We noticed that measuring the energy consumption for DP-CTGAN resulted in a RAPL overflow, causing negative returns for the energy consumption during training. To prevent this, we inplemented a measurement within the DP-CTGAN source code. Starting from [line 118](https://github.com/PepijndeReus/PET-experiments/blob/main/dp_ctgan/dpctgan.py#L118) we measure the energy consumption per epoch in the source code instead of Phase2 of our scripts.

Details of the changes made are visible in [this commit](https://github.com/PepijndeReus/PET-experiments/commit/442ff87f7a501a3ed1fdb495f78d8dcb5689d566).