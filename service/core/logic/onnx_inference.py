import onnxruntime as rt
import cv2
import numpy as np

def emotions_detector(img_array):
    if len(img_array.shape)==2:
        img_array = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
    provider = ['CPUExecutionProvider']  # Adjust based on your hardware configuration
    m = rt.InferenceSession("modelViT.onnx", providers=provider)
    test_image = cv2.resize(img_array,(256,256))
    test_image = np.float32(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    
    onnx_pred = m.run(['dense'],{'input':test_image})
    emotion = ""
    if (np.argmax(onnx_pred[0][0])==0):
        emotion="angry"
    elif (np.argmax(onnx_pred[0][0])==1):
        emotion="happy"
    else:
        emotion = "sad"

    return {"Emotion":emotion}
