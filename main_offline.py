import os
import sys

os.system("nohup bash -c '" +
          sys.path[0] + "/main.py --size 192 >result.txt" +
          "' &")
