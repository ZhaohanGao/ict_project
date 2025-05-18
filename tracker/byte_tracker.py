import numpy as np
from .utils import STrack

class BYTETracker:
    def __init__(self, frame_rate=30):
        self.tracked_stracks = []
        self.frame_id = 0
        self.frame_rate = frame_rate

    def update(self, detections, img_size, ori_img_size):
        self.frame_id += 1
        activated_tracks = []

        new_stracks = [STrack(tlbr, cls_id=int(cls)) for *tlbr, score, cls in detections]

        for track in new_stracks:
            matched = False
            for existing in self.tracked_stracks:
                iou = self.iou(track.tlbr, existing.tlbr)
                if iou > 0.3:
                    existing.update(track)
                    activated_tracks.append(existing)
                    matched = True
                    break
            if not matched:
                activated_tracks.append(track)

        self.tracked_stracks = activated_tracks
        return self.tracked_stracks

    @staticmethod
    def iou(bb1, bb2):
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        inter_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        iou = inter_area / float(bb1_area + bb2_area - inter_area)
        return iou
