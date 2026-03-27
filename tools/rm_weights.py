import shutil
from pathlib import Path

weight_dirs = Path(r"C:\Pro\vedat\workdir")

for wfolder in weight_dirs.iterdir():
    if wfolder.is_dir():
        print(f"start to clear folder:{wfolder}")
        for wf in wfolder.iterdir():
            wf_nms= wf.name.split("_")
            if wf_nms[0]=="epoch" and wf_nms[1].isdigit():
                if int(wf_nms[1])<101:
                    print(f"rm file:{str(wf)}")
                    wf.unlink()