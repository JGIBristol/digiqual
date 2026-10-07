import sys

# Briefcase doesn't set sys.frozen. Setting it makes multiprocessing spawn
# workers as `<app exe> --multiprocessing-fork ...` (the app executable can't
# run `-c` code like python.exe can), which digiqual.gui.launcher.main() then
# routes to the worker code instead of opening another window.
sys.frozen = True

from digiqual.gui.launcher import main

if __name__ == "__main__":
    main()
