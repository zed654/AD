from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime


path = Path(__file__).resolve().parent.parent / "results"
writer = SummaryWriter(path)

print(
    f"\nTraining...\nRun tensorboard --logdir {path} and go to http://localhost:6006/ to see in real time the training performance.\n"
)

for i in range(100):
    writer.add_scalar('test_loss', i, global_step=i)

test_dict = {
    "test_key": 123,
    "test_key2": 456,
}
writer.add_scalar("4.test/1.test_reward", test_dict["test_key"], 101)
writer.close()


"""
    텐서보드 사용을 위해 tensorboard --logdir ../results 명령어 입력 필요
"""