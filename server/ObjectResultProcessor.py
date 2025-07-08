from client import TcpObjReceiver

class ObjectResultProcessor(TcpObjReceiver):
    def process_result(self):
        data = self.receve_data()

        # "class_id": cls_id,
        # "class_name": class_names[cls_id],
        # "confidence": round(conf, 3),
        # "bbox": [x1, y1, x2, y2]

        
