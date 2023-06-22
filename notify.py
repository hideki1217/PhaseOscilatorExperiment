from pathlib import Path

import utils

cwd = Path(__file__).absolute().parent

with open(cwd / "nohup.out") as f:
    stdout = '\n'.join(f.readlines())
utils.email_me(
    '[phase] Notification of end of execution',
    f"------- nohup.out -------\n{stdout}\n------------------")