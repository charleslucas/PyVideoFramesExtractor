import sys
import os
from os import walk
import os.path as osp
import argparse
import cv2
import math
from tqdm import tqdm
from multiprocessing import Pool

# TODO: add additional supported video formats
supported_video_ext = ('.avi', '.mp4')

# TODO: add additional supported extracted frame formats
supported_frame_ext = ('.jpg', '.png')


class FrameExtractor:
    def __init__(self, video_file, output_dir, frame_ext='.jpg', sampling=-1, tile_width=125, tile_height=125, stride=0):
        """Extract frames from video file and save them under a given output directory.

        Args:
            video_file (str)  : input video filename
            output_dir (str)  : output directory where video frames will be extracted
            frame_ext (str)   : extracted frame file format
            sampling (int)    : sampling rate -- extract one frame every given number of seconds.
                                Default=-1 for extracting all available frames
            tile_width (int)       : # of horizontal pixels per tile.
            tile_height (int)       : # of vertical pixels per tile.
            stride (int)      : # of horizontal or vertical pixels to shift across the original image per tile.
        """
        # Check if given video file exists -- abort otherwise
        if osp.exists(video_file):
            self.video_file = video_file
        else:
            raise FileExistsError('Video file {} does not exist.'.format(video_file))

        self.sampling = sampling
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.stride = stride

        # Create output directory for storing extracted frames
        self.output_dir = output_dir
        output_dir_root = output_dir
        output_dir_num = 0
        while osp.exists(self.output_dir):
            output_dir_num = output_dir_num + 1
            self.output_dir = "{}.{}".format(output_dir_root, output_dir_num)
        os.makedirs(self.output_dir)

        # Get extracted frame file format
        self.frame_ext = frame_ext
        if frame_ext not in supported_frame_ext:
            raise ValueError("Not supported frame file format: {}".format(frame_ext))
        else:
            self.frame_ext = frame_ext

        # Capture given video stream
        self.video = cv2.VideoCapture(self.video_file)

        # Get video fps
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)

        # Get video length in frames
        self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sampling != -1:
            self.video_length = self.video_length // self.sampling

        self.video_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.video_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Chop a video frame into possibly overlapping tiles offset by a given stride
    def chop_frame(self, frame_cnt, frame, video_width, video_height, tile_width, tile_height, stride):

        video_basename = osp.basename(self.video_file).split('.')[0]
        frame_rootname = "{}_{:08d}".format(video_basename, frame_cnt)

        print('video_width={}, video_height={}, stride={}'.format(video_width, video_height, stride))

        step_x = 0
        tile_count_x = 0
        while (step_x + tile_width) < video_width:
            step_y = 0
            tile_count_y = 0
            while (step_y + tile_height) < video_height:
                print('tile_count_x={}, tile_count_y={}, step_x={}, step_y={}, tile_width={}, tile_height={}'.format(tile_count_x, tile_count_y, step_x, step_y, tile_width, tile_height))
                cropped_image = frame[step_y:(step_y+tile_height),step_x:(step_x+tile_width)]
                curr_tile_filename = osp.join(self.output_dir, '{}_tile_x{:03d}_y{:03d}{}'.format(frame_rootname, tile_count_x, tile_count_y, self.frame_ext))
                print('crop_y = {}:{}, crop_x = {}:{}, curr_tile_filename={}'.format(step_y, (step_y+tile_height), step_x, (step_x+tile_width), curr_tile_filename))
                #cv2.imshow("tile", cropped_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cv2.imwrite(curr_tile_filename, cropped_image)
                step_y = step_y + stride
                tile_count_y = tile_count_y + 1
            step_x = step_x + stride
            tile_count_x = tile_count_x + 1

    def extract(self):
        # Get first frame
        success, frame = self.video.read()
        frame_cnt = 0
        while success:
            if self.stride == 0:
                # Write current frame as a single file
                video_basename = osp.basename(self.video_file).split('.')[0]
                curr_frame_filename = osp.join(self.output_dir, "{}_{:08d}{}".format(video_basename, frame_cnt, self.frame_ext))
                cv2.imwrite(curr_frame_filename, frame)
            else:
                self.chop_frame(frame_cnt, frame, self.video_width, self.video_height, self.tile_width, self.tile_height, self.stride)

            # Get next frame
            success, frame = self.video.read()

            if self.sampling != -1:
                frame_cnt += math.ceil(self.sampling * self.video_fps)
                self.video.set(1, frame_cnt)
            else:
                frame_cnt += 1

global args_


def extract_video_frames(v_file):
    global args_

    if os.stat(osp.join(args_.dir, v_file[0])).st_size > 0:
        if not osp.isdir(osp.join(args_.output_root, v_file[1])):
            # Set up video extractor for given video file
            extractor = FrameExtractor(video_file=osp.join(args_.dir, v_file[0]),
                                       output_dir=osp.join(args_.output_root, v_file[1]),
                                       sampling=args_.sampling,
                                       tile_width=args_.tile_width,
                                       tile_height=args_.tile_height,
                                       stride=args_.stride)
            # Extract frames
            extractor.extract()
    else:
        os.remove(osp.join(args_.dir, v_file[0]))


def check_sampling_param(s):
    s_ = float(s)
    if (s_ <= 0) and (s_ != -1):
        raise argparse.ArgumentTypeError("Please give a positive number of seconds or -1 for extracting all frames.")
    return s_


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Extract frames from videos")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--video', type=str, help='set video filename')
    group.add_argument('--dir', type=str, help='set videos directory')
    parser.add_argument('--sampling', type=check_sampling_param, default=-1,
                        help="extract 1 frame every args.sampling seconds (default: extract all frames)")
    parser.add_argument('--output-root', type=str, default='extracted_frames', help="set output root directory")
    parser.add_argument('--workers', type=int, default=None, help="Set number of multiprocessing workers")
    parser.add_argument('--tile_width', type=int, default=None, help="# of horizontal pixels per tile.")
    parser.add_argument('--tile_height', type=int, default=None, help="# of vertical pixels per tile.")
    parser.add_argument('--stride', type=int, default=0, help="# of horizontal or vertical pixels to shift across the original image per tile. (0=disabled)")
    args = parser.parse_args()

    # Extract frames from a (single) given video file
    if args.video:
        # Setup video extractor for given video file
        video_basename = osp.basename(args.video).split('.')[0]
        # Check video file extension
        video_ext = osp.splitext(args.video)[-1]
        if video_ext not in supported_video_ext:
            raise ValueError("Not supported video file format: {}".format(video_ext))
        # Set extracted frames output directory
        output_dir = osp.join(args.output_root, '{}_frames'.format(video_basename))
        # Set up video extractor for given video file
        extractor = FrameExtractor(video_file=args.video, output_dir=output_dir, sampling=args.sampling, tile_width=args.tile_width, tile_height=args.tile_height, stride=args.stride)
        # Extract frames
        extractor.extract()

    # Extract frames from all video files found under the given directory (including all sub-directories)
    if args.dir:
        print("#. Extract frames from videos under dir : {}".format(args.dir))
        print("#. Store extracted frames under         : {}".format(args.output_root))
        if args.sampling == -1:
            print("#. Extract all available frames.")
        else:
            print("#. Extract one frame every {} seconds.".format(args.sampling))
        print("#. Scan for video files...")

        # Scan given dir for video files
        video_list = []
        for r, d, f in walk(args.dir):
            for file in f:
                file_basename = osp.basename(file).split('.')[0]
                file_ext = osp.splitext(file)[-1]
                if file_ext in supported_video_ext:
                    video_list.append([osp.join(osp.relpath(r, args.dir), file),
                                       osp.join(osp.relpath(r, args.dir), "{}_frames".format(file_basename))])

        # Extract frames from found video files
        global args_
        args_ = args
        with Pool(processes=args.workers) as p:
            with tqdm(total=len(video_list)) as pbar:
                for i, _ in enumerate(p.imap_unordered(extract_video_frames, video_list)):
                    pbar.update()


if __name__ == '__main__':
    main()
