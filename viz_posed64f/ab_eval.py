"""eval_scene.py pointed at the A/B tree (ab/data, ab/out).  python ab_eval.py --scene bcd2436daf"""
import os, sys
sys.path.insert(0, "/scratch/ducpham/Working/spatial_reasoning/boxer/scannetpp_probe")   # eval_scene stays in the probe folder
import eval_scene
eval_scene.HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab")
eval_scene.main()
