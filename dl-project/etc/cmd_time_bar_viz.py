from time import sleep
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

# 예제 리스트 (10개의 가짜 작업)
files = [f"file_{i}" for i in range(10)]
data_count = 0

with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TimeElapsedColumn(),
    TextColumn("[green]{task.fields[arg1_n]}: [bold yellow]{task.fields[arg1]}"),
) as progress:
    task = progress.add_task(
        "[cyan]Processing files", total=len(files),
        arg1_n="Clear Task", arg1=0  # ← 여기서 커스텀 필드 arg1 설정
    )

    for f in files:
        sleep(0.3)  # 처리하는 척
        data_count += 50  # 가령 50개씩 처리했다고 가정

        progress.update(task, advance=1, arg1=data_count)  # 진행 + 표시 갱신