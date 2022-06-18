#!/usr/bin/env python3

if __name__ == '__main__':
    import sys
    from pathlib import Path
    resultdir, obsnum = sys.argv[1:]
    redu_dirs = list(Path(resultdir).glob('redu??'))
    if not redu_dirs:
        print('None')
        sys.exit(1)
    redu_dirs = sorted(
            redu_dirs, key=lambda x: int(x.name.replace('redu', '')))
    print(redu_dirs[-1].as_posix())
