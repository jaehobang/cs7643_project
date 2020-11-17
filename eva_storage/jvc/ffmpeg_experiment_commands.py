import subprocess
import numpy as np



class FfmpegExperimentCommands:



    @staticmethod
    def bitrate(images, save_directory, gop, **kwargs):
        """
        In this function, we will write the videos


        :param images:
        :param save_directory:
        :param kwargs:
        :return:
        """
        framerate = kwargs.get('framerate', 60)  ## default value is 60 if user doesn't provide it
        length, height, width, channels = images.shape
        arg_string = f'ffmpeg -f rawvideo -pix_fmt rgb24 -r {framerate} -s {width}x{height} -i pipe: -pix_fmt yuv420p -g {gop} -vcodec libx264 {save_directory} -y'
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
    def gop(images, save_directory, gop, **kwargs):
        """
        In this function, we will write the videos


        :param images:
        :param save_directory:
        :param kwargs:
        :return:
        """
        framerate = kwargs.get('framerate', 60)  ## default value is 60 if user doesn't provide it
        length, height, width, channels = images.shape
        arg_string = f'ffmpeg -f rawvideo -pix_fmt rgb24 -r {framerate} -s {width}x{height} -i pipe: -pix_fmt yuv420p -g {gop} -vcodec libx264 {save_directory} -y'
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