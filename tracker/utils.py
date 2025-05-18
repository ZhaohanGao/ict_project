class STrack:
    def __init__(self, tlbr, cls_id):
        self.tlbr = tlbr  # (x1, y1, x2, y2)
        self.class_id = cls_id
        self.track_id = STrack.next_id()
        self.tlwh = self._tlbr_to_tlwh(tlbr)

    def update(self, other):
        self.tlbr = other.tlbr
        self.tlwh = self._tlbr_to_tlwh(other.tlbr)

    @staticmethod
    def _tlbr_to_tlwh(tlbr):
        x1, y1, x2, y2 = tlbr
        return (x1, y1, x2 - x1, y2 - y1)

    _id_counter = 0

    @staticmethod
    def next_id():
        STrack._id_counter += 1
        return STrack._id_counter
