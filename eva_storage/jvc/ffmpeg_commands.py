"""
In this file, we will write the commands we will use to communicate with ffmpeg through python

@Jaeho Bang
"""

import os
import subprocess
import json
import numpy as np


class FfmpegCommands:


    @staticmethod
    def load_video(video_directory):
        ffprobe_command = f"ffprobe -v error -show_entries stream=width,height {video_directory}"
        ffprobe_args = ffprobe_command.split(' ')
        p = subprocess.Popen(ffprobe_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        output = out.decode('utf-8')
        tmp = output.split('\n')
        width = int(tmp[1].split('=')[1])
        height = int(tmp[2].split('=')[1])


        command = f"ffmpeg -i {video_directory} -vsync 0 -f rawvideo -pix_fmt rgb24 pipe:"
        args = command.split(" ")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        ##### time it takes to fetch the data

        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

        return video


    @staticmethod
    def ffprobe(video_directory):
        """
        TODO: For some reason, this function is not working, but it's not important right now
              It's something to fix later 8/13/2020

        :param video_directory:
        :return:
        """
        ## this function is currently not working but normal ffmpeg does work
        arg_string = f"ffprobe -select_streams v -of json {video_directory}"
        args = arg_string.split(' ')
        print(f"command: {arg_string}")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        print(out)
        final_output = json.loads(out.decode('utf-8'))
        print(final_output)
        return final_output


    @staticmethod
    def write_video(images, save_directory, **kwargs):
        framerate = kwargs.get('framerate', 60) ## default value is 60 if user doesn't provide it
        length, height, width, channels = images.shape
        arg_string = f'ffmpeg -f rawvideo -pix_fmt rgb24 -r {framerate} -s {width}x{height} -i pipe: -pix_fmt yuv420p -vcodec libx264 {save_directory} -y'
        print(f"final commmand: {arg_string}")
        args = arg_string.split(' ')
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}

        for frame in images:
            p.stdin.write(
                frame
                    .astype(np.uint8)
                    .tobytes()
            )
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        print(f"Wrote video to {save_directory}... Done!")
        return



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
        length, height, width, channels = images.shape
        #### force the key frames
        timestamps_str = ",".join(timestamps_list)
        #print(f"printing final timestamp str: {timestamps_str}")
        arg_string = f'ffmpeg -f rawvideo -pix_fmt rgb24 -r {framerate} -s {width}x{height} -i pipe: -pix_fmt yuv420p -vcodec libx264 -force_key_frames {timestamps_str} {save_directory} -y'
        args = arg_string.split(' ')

        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}

        for frame in images:
            p.stdin.write(
                frame
                    .astype(np.uint8)
                    .tobytes()
            )
        out, err = p.communicate(**communicate_kwargs)
        #p.stdin.close()
        #p.wait()

        ### not only do we have to do this, we have to literally write frame by frame...
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



    @staticmethod
    def get_iframes(video_directory):
        """
        In this function, we retrieve the actual content of the i frames using the -skip_frame command
        ## TODO: how do I actually do this? -- okay I think I can do this -- the command is below
        ffmpeg -i {video_directory} -f rawvideo -pix_fmt rgb24 pipe:

        :param video_directory: path to video
        :return: numpy array of all i frames
        """
        ###first we need to know the video's height and width
        ffprobe_command = f"ffprobe -v error -show_entries stream=width,height {video_directory}"
        ffprobe_args = ffprobe_command.split(' ')
        p = subprocess.Popen(ffprobe_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        output = out.decode('utf-8')
        tmp = output.split('\n')
        width = int(tmp[1].split('=')[1])
        height = int(tmp[2].split('=')[1])

        command = f"ffmpeg -discard nokey -i {video_directory} -vsync 0 -f rawvideo -pix_fmt rgb24 pipe:"
        ## TODO: we need to modify this command to include skip_frames

        args = command.split(" ")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        ##### time it takes to fetch the data

        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

        return video



    @staticmethod
    def get_iframe_indices(video_directory):
        command = f"ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv {video_directory}"
        print(f"Command: {command}")
        args = command.split(" ")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        final_output = out.decode('utf-8') ## I assume this is a string that have the number and \n
        ## we need to parse the final_output as a list of array
        final_output = final_output.split('\n')
        indices = [i for i,x in enumerate(final_output) if x == 'frame,I']
        return np.array(indices)


