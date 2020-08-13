"""
In this file, we will write the commands we will use to communicate with ffmpeg through python

@Jaeho Bang
"""

import os
import subprocess
import json



class FfmpegCommands:

    @staticmethod
    def force_keyframes(images, timestamps_list, save_directory, **kwargs):
        """

        we expect images to be a images in the form of numpy
        we expect key_indices to be in the form of indexid list
        ex: ffmpeg -i test3.mp4 -force_key_frames 0:00:00.05,0:00:00.10 test4.mp4
        this means we force an i frame at 0.05 sec, 0.10 sec

        :param images:
        :param timestamps_list: list of timestamps that will be used to force the i frames
        :return:
        """
        framerate = 60 if not kwargs['framerate'] else kwargs['framerate']
        length, width, height, channels = images.shape
        #### force the key frames
        timestamps_str = ",".join(timestamps_list)
        arg_string = f'ffmpeg -f rawvideo -pix_fmt rgb24 -r {framerate} -s {width}x{height} -i pipe: -pix_fmt yuv420p -vcodec libx264 -force_key_frames {timestamps_str} {save_directory} -y'
        args = arg_string.split(' ')

        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        print(f"Wrote video to {save_directory}... Done!")
        return

    @staticmethod
    def get_frameinfo(video_directory):
        """
        Returns indices of i-frames

        :param video_directory: path to video
        :return: list
        """
        assert(os.path.exists(video_directory))
        args = ['ffprobe', '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=pkt_pts_time,pict_type', '-of', 'json',
                video_directory]

        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        final_output = json.loads(out.decode('utf-8'))
        return final_output
