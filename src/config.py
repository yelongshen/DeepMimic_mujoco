from os import getcwd
import os

class Config(object):
    all_motions = ['backflip', 'cartwheel', 'crawl', 'dance_a', 'dance_b', 'getup_facedown'
                   'getup_faceup', 'jump', 'kick', 'punch', 'roll', 'run', 'spin', 'spinkick',
                   'walk']
    
    # Get the directory where config.py is located (src/)
    curr_path = os.path.dirname(os.path.abspath(__file__))
    
    # motion = 'spinkick'
    motion = 'dance_a'
    env_name = "dp_env_v3"

    motion_folder = '/deepmimic_mujoco/motions'
    xml_folder = '/deepmimic_mujoco/humanoid_deepmimic/envs/asset'
    xml_test_folder = '/mujoco_test/'

    mocap_path = "%s%s/humanoid3d_%s.txt"%(curr_path, motion_folder, motion)
    xml_path = "%s%s/%s.xml"%(curr_path, xml_folder, env_name)
    xml_path_test = "%s%s/%s_test.xml"%(curr_path, xml_test_folder, env_name)