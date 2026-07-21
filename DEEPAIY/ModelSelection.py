from DeepAlgF.location.location_l import Location_j
from DeepAlgF.ocr.location_ocr import Location_j_ocr
from DeepAlgF.ocr.transfer_learn import ocr_transfer_learn
from DeepAlgF.location.transfer_learn import loc_transfer_learn
from DeepAlgE.ultralytics.SegmentLabel import Segment_LZ
from DeepPositive.PositiveData import Positive_LZ


def parse_model(dict_data):
    bsw_DeepModeType = dict_data["bsw_DeepModeType"]
    if bsw_DeepModeType == "BSW_LOCATION_TYPE":
        loc_transfer_learn(dict_data)
    elif bsw_DeepModeType == "BSW_OCR_TYPE":
        ocr_transfer_learn(dict_data)
    elif bsw_DeepModeType == "BSW_SEGMENT_TYPE":
        seg = Segment_LZ()
        seg.Segment_json(dict_data)
    elif bsw_DeepModeType == "BSW_POSITIVE_TYPE":   # Positive
        pos = Positive_LZ()
        pos.cfg_param(dict_data)
    else:
        print("数据传输错误，请重新选择数据进行传输！")

class SelectModel():
    def __init__(self, json_s):
        self.json_s = json_s
        self.C_json()

    def C_json(self):
        # temp_json = json.loads(temp_json_s)
        temp_json = self.json_s
        cur_process = temp_json["bsw_DeepModeType"]
        if cur_process == "BSW_LOCATION_TYPE":
            loc = Location_j()
            loc.location_json(temp_json)
        elif cur_process == "BSW_OCR_TYPE":
            loc_ocr = Location_j_ocr()
            loc_ocr.location_json(temp_json)
        elif cur_process == "BSW_SEGMENT_TYPE":
            seg = Segment_LZ()
            seg.Segment_json(temp_json)
        elif cur_process == "BSW_POSITIVE_TYPE":
            pos = Positive_LZ()
            pos.cfg_param(temp_json)
        else:
            print("数据传输错误，请重新选择数据进行传输！")