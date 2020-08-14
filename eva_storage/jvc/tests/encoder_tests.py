from eva_storage.jvc.ffmpeg_commands import FfmpegCommands
from eva_storage.jvc.preprocessor import Preprocessor
from loaders.seattle_loader import SeattleLoader

import os


def test1():
    original_seattle_video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mov'

    original_save_dir = os.path.join(os.path.dirname(original_seattle_video_directory), 'seattle2_15000.mp4')
    save_dir = os.path.join(os.path.dirname(original_seattle_video_directory), 'seattle2_15000_jvc.mp4')

    original_frame_info = FfmpegCommands.get_frameinfo(original_save_dir)
    frame_info = FfmpegCommands.get_frameinfo(save_dir)
    i_count_old = 0
    i_count_new = 0

    for i, val in enumerate(original_frame_info['frames']):
        if val['pict_type'] == 'I':
            i_count_old += 1

    print(f"old: {i_count_old}")

    for i, val in enumerate(frame_info['frames']):
        if val['pict_type'] == 'I':
            i_count_new += 1

    print(f"new: {i_count_new}")

def test2():
    original_seattle_video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mov'

    original_save_dir = os.path.join(os.path.dirname(original_seattle_video_directory), 'seattle2_15000.mp4')
    save_dir = os.path.join(os.path.dirname(original_seattle_video_directory), 'seattle2_15000_jvc.mp4')
    frame_info = FfmpegCommands.get_frameinfo(save_dir)

    preprocessor = Preprocessor()
    video_filename = os.path.basename(save_dir)
    ###TODO: eliminate the extension
    video_filename = video_filename.split('.')[0]

    loader = SeattleLoader()
    tmp = loader.load_images(original_save_dir)

    rep_indices = preprocessor.run(tmp, video_filename)

    wrong_count = 0
    i_count_not_designated = 0
    for i, val in enumerate(frame_info['frames']):
        if i in rep_indices:
            if frame_info['frames'][i]['pict_type'] != 'I':
                print(f"frameid: {i} is wrong type (should be an I frame but it isn't)")
                wrong_count += 1
        if frame_info['frames'][i]['pict_type'] == 'I':
            if i not in rep_indices:
                i_count_not_designated += 1

    print(f"{wrong_count} {i_count_not_designated}")
    assert(wrong_count == 0)



if __name__ == "__main__":
    """
    Things we need to check for the encoder:
    1. What is the size comparison to the original video
    2. Is the i-frames we designate successfully converted in the new video?
    """
    test1()
    test2()

