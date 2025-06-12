## Running the experiment
`nohup python main.py > log_name.log 2>&1 &`
`nohup`: Runs the command immune to hangups (so it keeps running even if the terminal closes).
You can pick from the following flags as defined [in main.py](https://github.com/PepijndeReus/PET-experiments/blob/main/main.py#L14):
| Flag                | Description                                                                                                                                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-d`, `--dataset`   | Select one or more datasets to run on. Options: `student`, `breast`, `heart`, `census`. Defaults to all.                                                                                                                                      |
| `-s`, `--generator` | Choose one or more synthetic data generators. Options: `benchmark`, `dpctgan`, `ydata`, `ctgan`, `synthcity`, `datasynthesizer`, `nbsynthetic` *(note: `nbsynthetic` is disabled by default due to instability — see the paper for details)*. |
| `-p`, `--phases`    | Specify which experiment phases to run. Choose from phases `1` to `5`. Default: all phases.                                                                                                                                                   |
| `-t`, `--ml_task`   | Choose one or more machine learning tasks. Options: `knn` (K-Nearest Neighbors), `lr` (Logistic Regression), `nn` (Neural Network). Default: all tasks.                                                                                       |
| `-n`, `--amount`    | Number of synthetic datasets to generate per setting. Default: `5`.                                                                                                                                                                           |
| `-v`, `--verbose`   | Enable verbose mode for detailed logging. Enabled by default.                                                                                                                                                                                 |
| `-c`, `--clear`     | If set, clears the contents of the `measurements/` folder (deletes or resets .csv files before running).                                                                                                                                      |
| `--version`         | Show program version and exit. Currently: `0.1`.                                                                                                                                                                                              |

`>`: redirects stdout to the file.
`2>&1`: Redirects stderr (file descriptor 2) to the same place as stdout (file descriptor 1).
`&`: Sends the process to the background (so you can run new commands in the terminal if you wish.)